import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.distributions.normal import Normal
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models import informer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, start_idx=0):
        x = x + self.pe[start_idx:start_idx+x.size(0), :].unsqueeze(1)
        return self.dropout(x)

    
class ARTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, device, is_signature=False
        ):
        super(ARTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.is_signature = is_signature
        self.use_covariate_var_model = False

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.warm_start = self.kernel_size * 5

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            in_channels = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            self.conv_feats = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=in_channels, out_channels=nkernel,
                bias=False,
                #padding=self.kernel_size//2
            )

        self.conv_data = nn.Conv1d(
            kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            #bias=False,
            #padding=self.kernel_size//2
        )
        self.data_dropout = nn.Dropout(p=0.2)

        if self.use_feats:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(2*nkernel, nkernel, bias=False))
        else:
            self.linearMap = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, nkernel, bias=False))
        self.positional = PositionalEncoding(d_model=nkernel)

        enc_input_size = nkernel

        if self.t2v_type:
            if self.t2v_type not in ['local']:  
                self.t_size = sum([1 for (_, s) in self.feats_info.values() if s==-1])
            else:
                self.t_size = 1
            if self.t2v_type in ['mdh_lincomb']:
                self.t2v_layer_list = []
                for i in range(self.t_size):
                    self.t2v_layer_list.append(nn.Linear(1, nkernel))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                if self.t_size > 1:
                    self.t2v_linear =  nn.Linear(self.t_size*nkernel, nkernel)
                else:
                    self.t2v_linear = None
            elif self.t2v_type in ['local', 'mdh_parti', 'idx']:
                self.part_sizes = [nkernel//self.t_size]*self.t_size
                for i in range(nkernel%self.t_size):
                    self.part_sizes[i] += 1
                self.t2v_layer_list = []
                for i in range(len(self.part_sizes)):
                    self.t2v_layer_list.append(nn.Linear(1, self.part_sizes[i]))
                self.t2v_layer_list = nn.ModuleList(self.t2v_layer_list)
                self.t2v_dropout = nn.Dropout(p=0.2)
                self.t2v_linear = None
            #import ipdb ; ipdb.set_trace()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=nkernel, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['bivariate']:
            self.decoder_bv = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))
        if self.estimate_type in ['bivariate']:
            self.rho_layer = nn.Linear(nkernel, 2)

    def apply_signature(self, mean, X_in, feats_out, X_out):
        X_out = X_out - mean
        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)

            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_out.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_out.shape[0], self.kernel_size-1, X_out.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_out
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_out_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+X_out.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_out.shape[1], X_out.shape[0]).unsqueeze(-1)
                t_in = t_in / X_out.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            #if self.t2v_type is 'mdh_lincomb':
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            #import ipdb ; ipdb.set_trace()
            #t2v = torch.cat([t2v[0:1], torch.sin(t2v[1:])], dim=0)
            #enc_input = self.data_dropout(enc_input) + self.t2v_dropout(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)
        encoder_output = encoder_output.transpose(0, 1)

        return encoder_output

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=None
    ):

        #X_in = X_in[..., -X_in.shape[1]//5:, :]
        #feats_in = feats_in[..., -feats_in.shape[1]//5:, :]

        mean = X_in.mean(dim=1, keepdim=True)
        #std = X_in.std(dim=1,keepdim=True)
        X_in = (X_in - mean)

        #import ipdb ; ipdb.set_trace()
        if self.use_feats:
            feats_in_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)

            feats_in_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_in_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_in_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1],:].clamp(min=0)

        X_in_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        (X_in.shape[0], self.kernel_size-1, X_in.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    X_in
                ], dim=1
            ).transpose(1,2)
        ).transpose(1,2).clamp(min=0)#[..., :X_in.shape[1], :]

        if self.use_feats:
            enc_input = self.linearMap(torch.cat([feats_in_embed,X_in_embed],dim=-1)).transpose(0,1)
        else:
            enc_input = self.linearMap(X_in_embed).transpose(0,1)

        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(X_in.shape[1], X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_in[..., :, -self.t_size:].transpose(0,1)

            t2v = []
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            enc_input = enc_input + self.t2v_dropout(t2v)
        else:
            enc_input = self.positional(enc_input)
        encoder_output = self.encoder(enc_input)

        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_out_merged = torch.cat(
                [feats_in_merged[:,-self.warm_start+1:, :],feats_out_merged],
                dim=1
            )
            feats_out_embed = self.conv_feats(
                torch.cat(
                    [
                        torch.zeros(
                            (X_in.shape[0], self.kernel_size-1, feats_out_merged.shape[-1]),
                            dtype=torch.float, device=self.device
                        ),
                        feats_out_merged
                    ], dim=1
                ).transpose(1,2)
            ).transpose(1,2).clamp(min=0)

        #import ipdb ; ipdb.set_trace()
        X_out_embed = self.conv_data(
            torch.cat(
                [
                    torch.zeros(
                        [X_in.shape[0], self.kernel_size-1, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    ),
                    X_in[..., -self.warm_start+1:, :],
                    torch.zeros(
                        [X_in.shape[0], self.dec_len, X_in.shape[-1]],
                        dtype=torch.float, device=self.device
                    )
                ],
                dim=1
            ).transpose(1, 2)
        ).transpose(1, 2)

        if self.use_feats:
            dec_input = self.linearMap(torch.cat([feats_out_embed,X_out_embed],dim=-1)).transpose(0,1)
        else:
            dec_input = X_out_embed.transpose(0,1)
        #import ipdb ; ipdb.set_trace()
        if self.t2v_type:
            if self.t2v_type in ['local']:
                t_in = torch.arange(
                    X_in.shape[1], X_in.shape[1]+self.dec_len, dtype=torch.float, device=self.device
                ).unsqueeze(1).expand(self.dec_len, X_in.shape[0]).unsqueeze(-1)
                t_in = t_in / X_in.shape[1] * 10.
            else:
                t_in = feats_out[..., :, -self.t_size:].transpose(0,1)
            t2v = []
            if self.t2v_type in ['local', 'mdh_parti', 'idx', 'mdh_lincomb']:
                for i in range(self.t_size):
                    t2v_part = self.t2v_layer_list[i](t_in[..., :, i:i+1])
                    t2v_part = torch.cat([t2v_part[..., 0:1], torch.sin(t2v_part[..., 1:])], dim=-1)
                    t2v.append(t2v_part)
                t2v = torch.cat(t2v, dim=-1)
                if self.t2v_linear is not None:
                    t2v = self.t2v_linear(t2v)
            dec_input = dec_input + self.t2v_dropout(t2v)
        else:
            dec_input = self.positional(dec_input, start_idx=X_in.shape[1])
        #import ipdb ; ipdb.set_trace()

        decoder_output = self.decoder_mean(dec_input, encoder_output).clamp(min=0)
        decoder_output = decoder_output.transpose(0,1)
        mean_out = self.linear_mean(decoder_output)

        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            X_pred = self.decoder_std(dec_input, encoder_output).clamp(min=0)
            X_pred = X_pred.transpose(0,1)
            std_out = F.softplus(self.linear_std(X_pred))
            if self.estimate_type in ['covariance']:
                v_out = self.linear_v(X_pred)
            if self.estimate_type in ['bivariate']:
                X_pred = self.decoder_bv(dec_input, encoder_output)
                X_pred = X_pred.transpose(0,1)
                rho_out = self.rho_layer(X_pred)
                rho_out = rho_out[..., -self.dec_len:, :]
                rho_1, rho_2 = rho_out[..., 1:, :], rho_out[..., :-1, :]
                #rho_out = torch.einsum("ijk,ijk->ij", (rho_1, rho_2)).unsqueeze(-1)
                rho_out = (rho_1 * rho_2).sum(dim=-1, keepdims=True)
                #import ipdb ; ipdb.set_trace()
                rho_out = torch.tanh(rho_out)
            #import ipdb ; ipdb.set_trace()

        mean_out = mean_out + mean

        if self.is_signature:
            signature_state = self.apply_signature(mean, X_in, feats_out, X_out)
            decoder_output = decoder_output[..., -self.dec_len:, :]

        #import ipdb ; ipdb.set_trace()

        if self.is_signature:
            if self.estimate_type in ['point']:
                return mean_out[..., -self.dec_len:, :], decoder_output, signature_state
            elif self.estimate_type in ['variance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], decoder_output, signature_state)
            elif self.estimate_type in ['covariance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :], decoder_output, signature_state)
            elif self.estimate_type in ['bivariate']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out, decoder_output, signature_state)
        else:
            if self.estimate_type in ['point']:
                return mean_out[..., -self.dec_len:, :]
            elif self.estimate_type in ['variance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :])
            elif self.estimate_type in ['covariance']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], v_out[..., -self.dec_len:, :])
            elif self.estimate_type in ['bivariate']:
                return (mean_out[..., -self.dec_len:, :], std_out[..., -self.dec_len:, :], rho_out)


class GPTTransformerModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, kernel_size, nkernel, is_nar, device, is_signature=False
        ):
        super(GPTTransformerModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.device = device
        self.is_signature = is_signature
        self.d_transform_typ = 'conv'
        self.f_transform_typ = 'conv'
        self.is_nar = is_nar

        self.kernel_size = kernel_size
        self.nkernel = nkernel

        self.positional = PositionalEncoding(d_model=nkernel)
        if self.is_nar:
            self.warm_start = self.dec_len - self.kernel_size
        else:
            self.warm_start = 0

        if self.use_feats:
            self.use_local_weights = False
            self.embed_feat_layers = {}
            for idx, (card, emb_size) in self.feats_info.items():
                if card != -1 and card != 0 and emb_size > 0:
                    self.embed_feat_layers[str(idx)] = nn.Embedding(card, emb_size)
                elif emb_size == -2:
                    self.use_local_weights = True
                    self.tsid_idx = idx
                    self.num_local_weights = card
            self.embed_feat_layers = nn.ModuleDict(self.embed_feat_layers)
            feats_dim = sum([s for (_, s) in self.feats_info.values() if s>-1])

            if self.f_transform_typ in ['linear']:
                self.f_transform_lyr = nn.Linear(feats_dim, feats_dim)
            elif self.f_transform_typ in ['conv']:
                self.f_transform_lyr = nn.Conv1d(
                    kernel_size=self.kernel_size, stride=1,
                    in_channels=feats_dim, out_channels=feats_dim,
                )

            self.linear_map = nn.Linear(feats_dim+self.nkernel, self.nkernel)

        if self.d_transform_typ in ['linear']:
            self.d_transform_lyr = nn.Linear(1, nkernel)
        elif self.d_transform_typ in ['conv']:
            self.d_transform_lyr = nn.Conv1d(
                kernel_size=self.kernel_size, stride=1, in_channels=1, out_channels=nkernel,
            )

        enc_input_size = nkernel
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        dec_input_size = nkernel
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_input_size, nhead=4, dropout=0, dim_feedforward=512
        )
        self.decoder_mean = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            self.decoder_std = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        if self.estimate_type in ['bivariate']:
            self.decoder_bv = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        if self.use_local_weights:
            #self.linear_mean = []
            #for i in range(self.num_local_weights):
            #    self.linear_mean.append(nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1)))
            #self.linear_mean = nn.ModuleList(self.linear_mean)
            self.linear_mean = nn.Embedding(self.num_local_weights, nkernel)
        else:
            self.linear_mean = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['variance', 'covariance', 'bivariate']:
            if self.use_local_weights:
                #self.linear_std = []
                #for i in range(self.num_local_weights):
                #    self.linear_std.append(nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1)))
                #self.linear_std = nn.ModuleList(self.linear_std)
                self.linear_std = nn.Embedding(self.num_local_weights, nkernel)
            else:
                self.linear_std = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, 1))
        if self.estimate_type in ['covariance']:
            self.linear_v = nn.Sequential(nn.ReLU(), nn.Linear(nkernel, self.v_dim))
        if self.estimate_type in ['bivariate']:
            self.rho_layer = nn.Linear(nkernel, 2)

    def merge_feats(self, feats):
        feats_merged = []
        for idx, efl in self.embed_feat_layers.items():
            feats_i = feats[..., int(idx)].type(torch.LongTensor).to(self.device)
            feats_merged.append(efl(feats_i))
        for idx, (card, emb_size) in self.feats_info.items():
            if card == 0:
                feats_merged.append(feats[..., idx:idx+1])
        feats_merged = torch.cat(feats_merged, dim=2)
        return feats_merged

    def pad_for_conv(self, x, trf_type):
        if trf_type in ['conv']:
            x_padded = torch.cat(
                [
                    torch.zeros(
                        (x.shape[0], self.kernel_size-1, x.shape[2]),
                        dtype=torch.float, device=self.device
                    ),
                    x
                ],
                dim=1
            )
        elif trf_type in ['linear']:
            x_padded = x
        return x_padded

    def d_transform(self, x):
        if self.d_transform_typ in ['linear']:
            x_transform = self.d_transform_lyr(x)
        elif self.d_transform_typ in ['conv']:
            x_transform = self.d_transform_lyr(x.transpose(1,2)).transpose(1,2)

        return x_transform

    def f_transform(self, x):
        if self.f_transform_typ in ['linear']:
            x_transform = self.f_transform_lyr(x)
        elif self.f_transform_typ in ['conv']:
            x_transform = self.f_transform_lyr(x.transpose(1,2)).transpose(1,2)

        return x_transform

    def forward(
        self, feats_in, X_in, feats_out, X_out=None, teacher_force=None
    ):

        mean = X_in.mean(dim=1, keepdim=True)
        #mean, _ = X_in.min(dim=1, keepdim=True)
        X_in = (X_in - mean)

        X_in_transformed = self.d_transform(self.pad_for_conv(X_in, self.d_transform_typ))
        if self.use_feats:
            feats_in_merged = self.merge_feats(feats_in)
            feats_in_transformed = self.f_transform(
                self.pad_for_conv(feats_in_merged, self.f_transform_typ)
            )
            enc_input = self.linear_map(torch.cat([feats_in_transformed, X_in_transformed], dim=-1))
        else:
            enc_input = X_in_transformed
        encoder_output = self.encoder(self.positional(enc_input.transpose(0,1)))

        #import ipdb ; ipdb.set_trace()
        if self.d_transform_typ in ['linear']: ps = 1 + self.warm_start
        elif self.d_transform_typ in ['conv']: ps = self.kernel_size + self.warm_start
        lps = X_in.shape[1] - ps + int(self.is_nar)
        if self.use_feats:
            if self.f_transform_typ in ['linear']: f_ps = 1 + self.warm_start
            elif self.f_transform_typ in ['conv']: f_ps = self.kernel_size + self.warm_start
            lf_ps = X_in.shape[1] - f_ps + int(self.is_nar)

        if self.is_nar: out_len = self.dec_len
        else: out_len = self.dec_len - 1

        #if X_out is not None:
        if self.is_nar==True or X_out is not None:
            if self.is_nar==True:
                X_out = torch.zeros(
                    (X_in.shape[0], self.dec_len, X_in.shape[2]),
                    dtype=torch.float, device=self.device
                )
            X_out_padded = torch.cat([X_in[..., lps:, :], X_out[..., :out_len, :]], dim=1)
            X_out_transformed = self.d_transform(X_out_padded)
            if self.use_feats:
                feats_out_padded = torch.cat(
                    [feats_in[..., lf_ps:, :], feats_out[..., :out_len, :]],
                    dim=1
                )
                feats_out_merged = self.merge_feats(feats_out_padded)
                feats_out_transformed = self.f_transform(feats_out_merged)
                #import ipdb ; ipdb.set_trace()
                dec_input = self.linear_map(
                    torch.cat([feats_out_transformed, X_out_transformed], dim=-1)
                )
            else:
                dec_input = X_out_transformed
            #dec_input = torch.cat([enc_input[..., ps:, :], dec_input[..., :out_len, :]], dim=1)
            #dec_input = dec_input[..., :, :]
            dec_input = self.positional(dec_input.transpose(0,1))
            #import ipdb ; ipdb.set_trace()

            decoder_output = self.decoder_mean(dec_input, encoder_output)#.clamp(min=0)
            decoder_output = decoder_output.transpose(0,1)
            if self.use_local_weights:
                local_indices = feats_out[..., self.tsid_idx].type(torch.LongTensor).to(self.device)
                local_weights = self.linear_mean(local_indices)
                decoder_output = decoder_output[..., -self.dec_len:, :]
                #import ipdb ; ipdb.set_trace()
                mean_out = (decoder_output * local_weights).sum(-1, keepdims=True)
                #mean_out = torch.einsum('ijk,ilk->ij', (local_weights, decoder_output)).unsqueeze(-1)
            else:
                mean_out = self.linear_mean(decoder_output)

            if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                X_pred = self.decoder_std(dec_input, encoder_output)#.clamp(min=0)
                X_pred = X_pred.transpose(0,1)
                if self.use_local_weights:
                    local_indices = feats_out[..., self.tsid_idx].type(torch.LongTensor).to(self.device)
                    local_weights = self.linear_std(local_indices)
                    X_pred = X_pred[..., -self.dec_len:, :]
                    std_out = (X_pred * local_weights).sum(-1, keepdims=True)
                    #std_out = torch.einsum('ijk,ilk->ij', (local_weights, X_pred)).unsqueeze(-1)
                    std_out = F.softplus(std_out)
                else:
                    std_out = F.softplus(self.linear_std(X_pred))

        else:
            std_out = []
            #import ipdb ; ipdb.set_trace()
            mean_out = list(torch.split(X_in[..., -ps:, :], 1, dim=1))
            if self.use_feats:
                f_padded = torch.cat([feats_in[..., -f_ps:, :], feats_out[..., :out_len, :]], dim=1)
            for i in range(0, self.dec_len):
                x_i = torch.cat(mean_out[-ps:], dim=1)
                x_i_transformed = self.d_transform(x_i)
                if self.use_feats:
                    f_i = f_padded[..., i:i+f_ps, :]
                    f_i_merged = self.merge_feats(f_i)
                    f_i_transformed = self.f_transform(f_i_merged)
                    dec_input = self.linear_map(torch.cat([f_i_transformed, x_i_transformed], dim=-1))
                else:
                    dec_input = x_i_transformed
                dec_input = self.positional(dec_input.transpose(0,1), start_idx=i)
                #import ipdb ; ipdb.set_trace()

                decoder_output = self.decoder_mean(dec_input, encoder_output)#.clamp(min=0)
                decoder_output = decoder_output.transpose(0,1)
                if self.use_local_weights:
                    local_indices = f_padded[..., f_ps+i:f_ps+i+1, self.tsid_idx].type(torch.LongTensor).to(self.device)
                    local_weights = self.linear_mean(local_indices)
                    decoder_output = decoder_output[..., -self.dec_len:, :]
                    mean_out_ = (decoder_output * local_weights).sum(-1, keepdims=True)
                else:
                    mean_out_ = self.linear_mean(decoder_output)

                if self.estimate_type in ['variance', 'covariance', 'bivariate']:
                    X_pred = self.decoder_std(dec_input, encoder_output)#.clamp(min=0)
                    X_pred = X_pred.transpose(0,1)
                    if self.use_local_weights:
                        local_indices = f_padded[..., f_ps+i:f_ps+i+1, self.tsid_idx].type(torch.LongTensor).to(self.device)
                        local_weights = self.linear_std(local_indices)
                        X_pred = X_pred[..., -self.dec_len:, :]
                        std_out_ = (X_pred * local_weights).sum(-1, keepdims=True)
                        std_out_ = F.softplus(std_out_)
                    else:
                        std_out_ = F.softplus(self.linear_std(X_pred))

                mean_out.append(mean_out_)
                if self.estimate_type in ['variance']:
                    std_out.append(std_out_)

            #import ipdb ; ipdb.set_trace()
            mean_out = torch.cat(mean_out, 1)
            if self.estimate_type in ['variance']:
                std_out = torch.cat(std_out, 1)

        mean_out = mean_out + mean

        mean_out = mean_out[..., -self.dec_len:, :]
        if self.estimate_type in ['variance']:
            std_out = std_out[..., -self.dec_len:, :]

        if self.estimate_type in ['point']:
            return mean_out
        elif self.estimate_type in ['variance']:
            return (mean_out, std_out)


class OracleModel(nn.Module):
    def __init__(
            self, dec_len, estimate_type
            ):
        super(OracleModel, self).__init__()

        self.dec_len = dec_len
        self.estimate_type = estimate_type
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=None):
        assert X_out is not None
        dists = []
        for i in range(X_in.shape[1]-self.dec_len):
            dist = torch.pow(X_in[:, i:i+self.dec_len] - X_out, 2).mean(dim=1)
            #dist = torch.abs(X_in[:, i:i+self.dec_len] - X_out).mean(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=1)
        #import ipdb; ipdb.set_trace()
        min_indices_ = torch.argmin(dists, dim=1).unsqueeze(-1).unsqueeze(-1)
        min_indices = []
        for i in range(self.dec_len):
            min_indices.append(min_indices_+i)
        min_indices = torch.cat(min_indices, dim=1)
        mean_out = X_in.gather(1, min_indices)

        #import ipdb; ipdb.set_trace()

        return mean_out


class OracleForecastModel(nn.Module):
    def __init__(
            self, dec_len, estimate_type
            ):
        super(OracleForecastModel, self).__init__()

        self.dec_len = dec_len
        self.estimate_type = estimate_type
        self.dummy_layer = nn.Linear(1, 1)
        self.warm_start = self.dec_len * 2

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=None):
        dists = []
        key = X_in[:, -self.warm_start:]
        for i in range(X_in.shape[1]-2*self.warm_start):
            dist = torch.pow(X_in[:, i:i+self.warm_start] - key, 2).mean(dim=1)
            #dist = torch.abs(X_in[:, i:i+self.warm_start] - key).mean(dim=1)
            dists.append(dist)
        dists = torch.cat(dists, dim=1)
        #import ipdb; ipdb.set_trace()
        min_indices_ = torch.argmin(dists, dim=1).unsqueeze(-1).unsqueeze(-1)
        min_indices = []
        for i in range(self.dec_len):
            min_indices.append(min_indices_+self.warm_start+i)
        min_indices = torch.cat(min_indices, dim=1)
        mean_out = X_in.gather(1, min_indices)

        #import ipdb; ipdb.set_trace()

        return mean_out


class RNNNARModel(nn.Module):
    def __init__(
            self, dec_len, num_rnn_layers, feats_info, hidden_size, batch_size,
            estimate_type, use_feats, v_dim, device
        ):
        super(RNNNARModel, self).__init__()

        self.dec_len = dec_len
        self.num_rnn_layers = num_rnn_layers
        self.feats_info = feats_info
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.estimate_type = estimate_type
        self.device = device
        self.use_feats = use_feats
        self.v_dim = v_dim

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            feats_embed_dim = sum([s for (_, s) in self.feats_info.values() if s is not -1])
        else:
            feats_embed_dim = 0
        enc_input_size = 1 + feats_embed_dim
        self.encoder = nn.LSTM(enc_input_size, self.hidden_size, batch_first=True)

        if self.use_feats:
            dec_input_size = self.hidden_size + feats_embed_dim
        else:
            dec_input_size = self.hidden_size

        self.decoder_mean = nn.Sequential(
            nn.Linear(dec_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        if self.estimate_type in ['variance', 'covariance']:
            self.decoder_std = nn.Sequential(
                nn.Linear(dec_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Softplus()
            )

        if self.estimate_type in ['covariance']:
            self.decoder_v = nn.Sequential(
                nn.Linear(dec_input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.v_dim),
                #nn.Softplus()
            )

    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )

    def forward(self, feats_in, X_in, feats_out, X_out=None):

        if self.use_feats:
            feats_in_merged, feats_out_merged = [], []
            for i in range(feats_in.shape[-1]):
                card = self.feats_info[i][0]
                if card != -1:
                    if card != 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)
            for i in range(feats_out.shape[-1]):
                card = self.feats_info[i][0]
                if card != -1:
                    if card != 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
            feats_in_embed = feats_in_merged
            feats_out_embed = feats_out_merged
            enc_input = torch.cat([feats_in_embed, X_in], dim=-1)
        else:
            enc_input = X_in

        enc_hidden = self.init_hidden(X_in.shape[0])
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        enc_output_tile = enc_output[..., -1:, :].repeat(1, self.dec_len, 1)
        if self.use_feats:
            dec_input = torch.cat([feats_out_embed, enc_output_tile], dim=-1)
        else:
            dec_input = enc_output_tile
        means = self.decoder_mean(dec_input)
        if self.estimate_type in ['variance', 'covariance']:
            stds = self.decoder_std(dec_input)
        if self.estimate_type in ['covariance']:
            v = self.decoder_v(dec_input)

        if self.estimate_type in ['point']:
            return means
        elif self.estimate_type in ['variance']:
            return (means, stds)
        elif self.estimate_type in ['covariance']:
            return (means, stds, v)
        

class RNNARModel(nn.Module):
    def __init__(
            self, dec_len, feats_info, estimate_type, use_feats, t2v_type,
            v_dim, num_rnn_layers, hidden_size, batch_size, device, is_signature=False
        ):
        super(RNNARModel, self).__init__()

        self.dec_len = dec_len
        self.feats_info = feats_info
        self.estimate_type = estimate_type
        self.use_feats = use_feats
        self.t2v_type = t2v_type
        self.v_dim = v_dim
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.is_signature = is_signature

        if self.use_feats:
            self.embed_feat_layers = []
            for idx, (card, emb_size) in self.feats_info.items():
                if card is not -1:
                    if card is not 0:
                        self.embed_feat_layers.append(nn.Embedding(card, emb_size))
                    else:
                        self.embed_feat_layers.append(nn.Linear(1, 1, bias=False))
            self.embed_feat_layers = nn.ModuleList(self.embed_feat_layers)

            feats_embed_dim = sum([s for (_, s) in self.feats_info.values() if s is not -1])
            enc_input_size = 1 + feats_embed_dim
        else:
            enc_input_size = 1

        self.encoder = nn.LSTM(enc_input_size, self.hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(enc_input_size, self.hidden_size,  batch_first=True)
        self.decoder_mean = nn.Linear(hidden_size, 1)
        self.decoder_std = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())
        self.decoder_v = nn.Linear(hidden_size, self.v_dim)

    def init_hidden(self, batch_size):
        #[num_layers*num_directions,batch,hidden_size]   
        return (
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size, device=self.device)
        )

    def forward(self, feats_in, X_in, feats_out, X_out=None, teacher_force=True):

        if self.use_feats:
            feats_in_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_in_ = feats_in[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_in_ = feats_in[..., i:i+1]
                    feats_in_merged.append(
                        self.embed_feat_layers[i](feats_in_)
                    )
            feats_in_merged = torch.cat(feats_in_merged, dim=2)
        feats_in_embed = feats_in_merged

        enc_input = torch.cat([feats_in_embed, X_in], dim=-1)
        enc_hidden = self.init_hidden(X_in.shape[0])
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        if self.use_feats:
            feats_out_merged = []
            for i in range(len(self.feats_info)):
                card = self.feats_info[i][0]
                if card is not -1:
                    if card is not 0:
                        feats_out_ = feats_out[..., i].type(torch.LongTensor).to(self.device)
                    else:
                        feats_out_ = feats_out[..., i:i+1]
                    feats_out_merged.append(
                        self.embed_feat_layers[i](feats_out_)
                    )
            feats_out_merged = torch.cat(feats_out_merged, dim=2)
        feats_out_embed = feats_out_merged

        dec_state = enc_state
        if X_out is not None:
            X_prev = torch.cat([X_in[..., -1:, :], X_out[..., :-1, :]], dim=1)
            feats_prev = torch.cat([feats_in_embed[..., -1:, :], feats_out_embed[..., :-1, :]], dim=1)
            dec_input = torch.cat([feats_prev, X_prev], dim=-1)
            dec_output, dec_state = self.decoder_lstm(dec_input, dec_state)
            means = self.decoder_mean(dec_output)
            if self.estimate_type in ['covariance', 'variance']:
                stds = self.decoder_std(dec_output)
                if self.estimate_type in ['covariance']:
                    v = self.decoder_v(dec_output)
        else:
            X_prev = X_in[..., -1:, :]
            feats_prev = feats_in_embed[..., -1:, :]
            means, stds, v = [], [], []
            for i in range(self.dec_len):
                dec_input = torch.cat([feats_prev, X_prev], dim=-1)
                dec_output, dec_state = self.decoder_lstm(dec_input, dec_state)
                step_pred_mu = self.decoder_mean(dec_output)
                means.append(step_pred_mu)
                if self.estimate_type in ['covariance', 'variance']:
                    step_pred_std = self.decoder_std(dec_output)
                    stds.append(step_pred_std)
                    if self.estimate_type in ['covariance']:
                        step_pred_v = self.decoder_v(dec_output)
                        v.append(step_pred_v)
                X_prev = step_pred_mu
                feats_prev = feats_out_embed[..., i:i+1, :]

            means = torch.cat(means, dim=1)
            if self.estimate_type in ['covariance', 'variance']:
                stds = torch.cat(stds, dim=1)
                if self.estimate_type in ['covariance']:
                    v = torch.cat(v, dim=1)

        if self.estimate_type in ['point']:
            return means
        elif self.estimate_type in ['variance']:
            return means, stds
        elif self.estimate_type in ['covariance']:
            return means, stds, v


def get_base_model(
    args, base_model_name, level, N_input, N_output,
    input_size, output_size, estimate_type, feats_info
):

    #hidden_size = max(int(config['hidden_size']*1.0/int(np.sqrt(level))), args.fc_units)
    hidden_size = args.hidden_size

    if base_model_name in ['rnn-mse-nar', 'rnn-nll-nar', 'rnn-fnll-nar']:
        net_gru = RNNNARModel(
            dec_len=N_output,
            num_rnn_layers=args.num_grulstm_layers,
            feats_info=feats_info,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            estimate_type=estimate_type,
            use_feats=args.use_feats,
            v_dim=args.v_dim,
            device=args.device
        ).to(args.device)
    elif base_model_name in ['rnn-mse-ar', 'rnn-nll-ar', 'rnn-fnll-ar']:
        net_gru = RNNARModel(
            dec_len=N_output,
            feats_info=feats_info,
            estimate_type=estimate_type,
            use_feats=args.use_feats,
            t2v_type=args.t2v_type,
            v_dim=args.v_dim,
            num_rnn_layers=args.num_grulstm_layers,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            device=args.device
        ).to(args.device)
    elif base_model_name in ['trans-mse-ar', 'trans-nll-ar', 'trans-fnll-ar', 'trans-bvnll-ar']:
            net_gru = ARTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=10, nkernel=32, device=args.device
            ).to(args.device)
    elif base_model_name in ['gpt-nll-ar', 'gpt-mse-ar']:
            net_gru = GPTTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=args.kernel_size, nkernel=args.nkernel, is_nar=False, device=args.device
            ).to(args.device)
    elif base_model_name in ['gpt-nll-nar', 'gpt-mse-nar']:
            net_gru = GPTTransformerModel(
                N_output, feats_info, estimate_type, args.use_feats,
                args.t2v_type, args.v_dim,
                kernel_size=args.kernel_size, nkernel=args.nkernel, is_nar=True, device=args.device
            ).to(args.device)
    elif base_model_name in ['informer-mse-nar']:
            net_gru = informer.Informer(
                enc_in=1,
                dec_in=1,
                c_out=1,
                seq_len=N_input,
                label_len=N_output,
                out_len=N_output,
                factor=5,
                d_model=512,
                n_heads=8,
                e_layers=2,
                d_layers=1,
                d_ff=2048,
                dropout=0.05,
                attn='prob',
                embed='fixed',
                freq=args.freq,
                activation='gelu',
                output_attention=False,
                distil=True,
                mix=True,
                feats_info=feats_info,
                device=args.device
            ).to(args.device)
    elif base_model_name in ['transsig-nll-nar']:
        net_gru = ARTransformerModel(
            N_output, feats_info, estimate_type, args.use_feats,
            args.t2v_type, args.v_dim,
            kernel_size=10, nkernel=32, device=args.device,
            is_signature=True
        ).to(args.device)
    elif base_model_name in ['oracle']:
        net_gru = OracleModel(N_output, estimate_type).to(args.device)
    elif base_model_name in ['oracleforecast']:
        net_gru = OracleForecastModel(N_output, estimate_type).to(args.device)

    return net_gru
