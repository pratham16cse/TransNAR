from torch.utils.data import DataLoader
import torch
from torch.distributions.normal import Normal
import numpy as np
import os
from collections import OrderedDict
import pywt
import pandas as pd
import re
import time
import shutil
from tsmoothie.smoother import SpectralSmoother, ExponentialSmoother
from statsmodels.tsa.seasonal import seasonal_decompose
import time

from data.synthetic_dataset import create_synthetic_dataset, create_sin_dataset, SyntheticDataset
from data.real_dataset import parse_ett, parse_Solar, parse_etthourly, parse_aggtest, parse_electricity, parse_foodinflation, parse_foodinflationmonthly


to_float_tensor = lambda x: torch.FloatTensor(x.copy())
to_long_tensor = lambda x: torch.FloatTensor(x.copy())

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def clean_trial_checkpoints(result):
    for trl in result.trials:
        trl_paths = result.get_trial_checkpoints_paths(trl,'metric')
        for path, _ in trl_paths:
            shutil.rmtree(path)

def add_metrics_to_dict(
    metrics_dict, model_name, metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae,
        metric_smape
):
    #if model_name not in metrics_dict:
    #    metrics_dict[model_name] = dict()

    metrics_dict['mse'] = metric_mse
    metrics_dict['dtw'] = metric_dtw
    metrics_dict['tdi'] = metric_tdi
    metrics_dict['crps'] = metric_crps
    metrics_dict['mae'] = metric_mae
    metrics_dict['smape'] = metric_smape

    return metrics_dict

def add_base_metrics_to_dict(
    metrics_dict, agg_method, K, model_name, metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae,
):
    if agg_method not in metrics_dict:
        metrics_dict[agg_method] = {}
    if K not in metrics_dict[agg_method]:
        metrics_dict[agg_method][K] = {}
    if model_name not in metrics_dict[agg_method][K]:
        metrics_dict[agg_method][K][model_name] = {}

    metrics_dict[agg_method][K][model_name]['mse'] = metric_mse
    metrics_dict[agg_method][K][model_name]['dtw'] = metric_dtw
    metrics_dict[agg_method][K][model_name]['tdi'] = metric_tdi
    metrics_dict[agg_method][K][model_name]['crps'] = metric_crps
    metrics_dict[agg_method][K][model_name]['mae'] = metric_mae
    #metrics_dict[model_name]['smape'] = metric_smape
    
    return metrics_dict

def write_arr_to_file(
    output_dir, inf_model_name, inputs, targets, pred_mu, pred_std, pred_d, pred_v
):

    # Files are saved in .npy format
    np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_mu'), pred_mu)
    np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_std'), pred_std)
    np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_d'), pred_d)
    np.save(os.path.join(output_dir, inf_model_name + '_' + 'pred_v'), pred_v)

    for fname in os.listdir(output_dir):
        if fname.endswith('targets.npy'):
            break
    else:
        np.save(os.path.join(output_dir, 'inputs'), inputs)
        np.save(os.path.join(output_dir, 'targets'), targets)

def write_aggregate_preds_to_file(
    output_dir, base_model_name, agg_method, level, inputs, targets, pred_mu, pred_std
):

    # Files are saved in .npy format
    sep = '__'
    model_str = base_model_name + sep + agg_method + sep  + str(level)
    agg_str = agg_method + sep  + str(level)

    np.save(os.path.join(output_dir, model_str + sep + 'pred_mu'), pred_mu.detach().numpy())
    np.save(os.path.join(output_dir, model_str + sep + 'pred_std'), pred_std.detach().numpy())

    suffix = agg_str + sep + 'targets.npy'
    for fname in os.listdir(output_dir):
        if fname.endswith(suffix):
            break
    else:
        np.save(os.path.join(output_dir, agg_str + sep + 'inputs'), inputs.detach().numpy())
        np.save(os.path.join(output_dir, agg_str + sep + 'targets'), targets.detach().numpy())


class Normalizer(object):
    def __init__(self, data, norm_type):
        super(Normalizer, self).__init__()
        self.norm_type = norm_type
        self.N = len(data)
        if norm_type in ['same']:
            pass
        elif norm_type in ['zscore_per_series']:
            self.mean = map(lambda x: x.mean(0, keepdims=True), data) #data.mean(1, keepdims=True)
            self.std = map(lambda x: x.std(0, keepdims=True), data) #data.std(1, keepdims=True)
            #import ipdb ; ipdb.set_trace()
            self.mean = torch.stack(list(self.mean), dim=0)
            self.std = torch.stack(list(self.std), dim=0)
            self.std = self.std.clamp(min=1., max=None)
        elif norm_type in ['zeroshift_per_series']:
            self.first = map(lambda x: x[0:1], data) #data.mean(1, keepdims=True)
            self.std = map(lambda x: x.std(0, keepdims=True), data)
            #import ipdb ; ipdb.set_trace()
            self.first = torch.stack(list(self.first), dim=0)
            self.std = torch.stack(list(self.std), dim=0)
            self.std = self.std.clamp(min=1., max=None)
        elif norm_type in ['min_per_series']:
            self.first = map(lambda x: x.min(0, keepdims=True)[0], data)
            self.std = map(lambda x: x.std(0, keepdims=True), data)
            #import ipdb ; ipdb.set_trace()
            self.first = torch.stack(list(self.first), dim=0)
            self.std = torch.stack(list(self.std), dim=0)
            self.std = self.std.clamp(min=1., max=None)
        elif norm_type in ['log']:
            pass
        elif norm_type in ['gaussian_copula']:
            ns = data.shape[1] * 1.
            #self.delta = 1. / (4*np.power(ns, 0.25) * np.power(np.pi*np.log(ns), 0.5))
            self.delta = 1e-5
            data_sorted, indices = data.sort(1)
            data_sorted_uq = torch.unique(data_sorted, sorted=True, dim=-1)
            counts = torch.cat(
                [(data_sorted == data_sorted_uq[:, i:i+1]).sum(dim=1, keepdims=True) for i in range(data_sorted_uq.shape[1])],
                dim=1
            )
            #import ipdb; ipdb.set_trace()
            self.x = data_sorted_uq
            self.x = torch.cat([self.x, 1.1*data_sorted[..., -1:]], dim=1)
            self.y = torch.cumsum(counts, 1)*1./data.shape[1]
            self.y = self.y.clamp(self.delta, 1.0-self.delta)
            self.y = torch.cat([self.y, torch.ones((data.shape[0], 1))*self.delta], dim=1)
            self.m = (self.y[..., 1:] - self.y[..., :-1]) / (self.x[..., 1:] - self.x[..., :-1])
            self.m = torch.maximum(self.m, torch.ones_like(self.m)*1e-4)
            self.c = self.y[..., :-1]
            #import ipdb; ipdb.set_trace()

            
    def normalize(self, data, ids=None, is_var=False):
        if ids is None:
            ids = torch.arange(self.N)

        if self.norm_type in ['same']:
            data_norm = data
        elif self.norm_type in ['zscore_per_series']:
            if not is_var:
                data_norm = (data - self.mean[ids]) / self.std[ids]
            else:
                data_norm = data / self.std[ids]
        elif self.norm_type in ['zeroshift_per_series', 'min_per_series']:
            if not is_var:
                data_norm = (data - self.first[ids]) / self.std[ids]
            else:
                data_norm = data / self.std[ids]
        elif self.norm_type in ['log']:
            data_norm = torch.log(data)
        elif self.norm_type in ['gaussian_copula']:
            # Piecewise linear fit of CDF
            indices = torch.searchsorted(self.x[ids], data).clamp(0, self.x.shape[-1])
            m = torch.gather(self.m[ids], -1, indices)
            c = torch.gather(self.c[ids], -1, indices)
            x_prev = torch.gather(self.x[ids], -1, indices)
            data_norm = (data - x_prev) * m + c
            data_norm = data_norm.clamp(self.delta, 1.0-self.delta)
            #import ipdb; ipdb.set_trace()
            
            # ICDF in standard normal
            dist = Normal(0., 1.)
            data_norm = dist.icdf(data_norm)
            #import ipdb; ipdb.set_trace()

        return data_norm.unsqueeze(-1)

    def unnormalize(self, data, ids=None, is_var=False):
        #return data # TODO Watch this
        if ids is None:
            ids = torch.arange(self.N)
        if self.norm_type in ['same']:
            data_unnorm = data
        elif self.norm_type in ['log']:
            data_unnorm = torch.exp(data)
        elif self.norm_type in ['zscore_per_series']:
            if not is_var:
                data_unnorm = data * self.std[ids] + self.mean[ids]
            else:
                data_unnorm = data * self.std[ids]
        elif self.norm_type in ['zeroshift_per_series', 'min_per_series']:
            if not is_var:
                data_unnorm = data * self.std[ids] + self.first[ids] 
            else:
                data_unnorm = data * self.std[ids]
        elif self.norm_type in ['gaussian_copula']:
            # CDF in standard normal
            dist = Normal(0., 1.)
            data = dist.cdf(data)

            # Inverse piecewise linear fit of CDF
            indices = torch.searchsorted(self.y[ids], data).clamp(0, self.x.shape[-1])
            m = torch.gather(self.m[ids], -1, indices)
            c = torch.gather(self.c[ids], -1, indices)
            x_prev = torch.gather(self.x[ids], -1, indices)
            data_unnorm = (data - c) / m + x_prev

        return data_unnorm

sqz = lambda x: np.squeeze(x, axis=-1)
expand = lambda x: np.expand_dims(x, axis=-1)

def get_a(agg_type, K):

    if K == 1:
        return torch.ones(1, dtype=torch.float)

    if agg_type in ['sum']:
        a = 1./K * torch.ones(K)
    elif agg_type in ['slope']:
        x = torch.arange(K, dtype=torch.float)
        m_x = x.mean()
        s_xx = ((x-m_x)**2).sum()
        a = (x - m_x) / s_xx
    elif agg_type in ['diff']:
        l = K // 2
        a_ = torch.ones(K)
        a = 1./K * torch.cat([-1.*a_[:l], a_[l:]], dim=0)
    return a

def aggregate_window(y, a, is_var, v=None):
    if is_var == False:
        y_a = (a*y).sum(dim=1, keepdims=True)
    else:
        w_d = (a**2*y).sum(dim=1, keepdims=True)
        if v is not None:
            #w_v = (((a.unsqueeze(-1)*v).sum(-1)**2)).sum(dim=1, keepdims=True)
            #av = a.unsqueeze(-1)*v
            #av = torch.matmul(av, av.transpose(-2,-1))
            #w_v = (((av).sum(-1)**2)).sum(dim=1, keepdims=True)
            w_v = (((a.unsqueeze(-1)*v)**2).sum(-1)).sum(dim=1, keepdims=True)
            y_a = w_d + w_v
        else:
            y_a = w_d

    return y_a

def aggregate_data(y, agg_type, K, is_var, a=None, v=None):
    # y shape: batch_size x N
    # if a need not be recomputed in every call, pass a vector directly
    # if v is not None, it is used as a V vector of low-rank multivariate gaussian
    # v shape: batch_size x N x args.v_dim
    bs, N = y.shape[0], y.shape[1]
    if a is None:
        a = get_a(agg_type, K)
    a = a.unsqueeze(0).repeat(bs, 1)
    y_agg = []
    for i in range(0, N, K):
        y_w = y[..., i:i+K]
        if v is not None:
            v_w = v[..., i:i+K, :]
            y_a = aggregate_window(y_w, a, is_var, v=v_w)
        else:
            y_a = aggregate_window(y_w, a, is_var)
        y_agg.append(y_a)
    y_agg = torch.cat(y_agg, dim=1)#.unsqueeze(-1)
    return y_agg


class TimeSeriesDataset(torch.utils.data.Dataset):
    """docstring for TimeSeriesDataset"""
    def __init__(
        self, data, enc_len, dec_len, feats_info, which_split,
        tsid_map=None, input_norm=None, target_norm=None,
        norm_type=None, feats_norms=None, train_obj=None
    ):
        super(TimeSeriesDataset, self).__init__()

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.which_split = which_split
        self.input_norm = input_norm
        self.target_norm = target_norm
        self.norm_type = norm_type
        self.feats_info = feats_info
        self.tsid_map = tsid_map
        self.feats_norms = feats_norms
        #self.train_obj = train_obj

        st = time.time()
        data_agg = []
        for i in range(0, len(data)):
            #print(i, len(data))
            ex = data[i]['target']
            ex_f = data[i]['feats']
            ex_len = len(ex)

            data_agg.append(
                {
                    'target':ex,
                    'feats':ex_f,
                }
            )

        et = time.time()
        print(which_split, 'total time:', et-st)

        if self.input_norm is None:
            assert norm_type is not None
            data_for_norm = []
            for i in range(0, len(data)):
                ex = data_agg[i]['target']
                data_for_norm.append(torch.FloatTensor(ex))
            #data_for_norm = to_float_tensor(data_for_norm).squeeze(-1)

            self.input_norm = Normalizer(data_for_norm, norm_type=self.norm_type)
            self.target_norm = self.input_norm
            del data_for_norm

            self.feats_norms = {}
            for j in range(len(self.feats_info)):
                card = self.feats_info[j][0]
                if card == 0:
                    feat_for_norm = []
                    for i in range(0, len(data)):
                        ex = data_agg[i]['feats'][:, j]
                        feat_for_norm.append(torch.FloatTensor(ex))
                    f_norm = Normalizer(feat_for_norm, norm_type='zscore_per_series')
                    self.feats_norms[j] = f_norm

        self.data = data_agg
        self.indices = []
        for i in range(0, len(self.data)):
            if which_split in ['train']:
                j = 0
                while j < len(self.data[i]['target']):
                    if j + self.enc_len + self.dec_len <= len(self.data[i]['target']):
                        self.indices.append((i, j))
                    j += 1
            elif which_split == 'dev':
                j = len(self.data[i]['target']) - self.enc_len - self.dec_len
                self.indices.append((i, j))
            elif which_split == 'test':
                j = len(self.data[i]['target']) - self.enc_len - self.dec_len
                self.indices.append((i, j))

    @property
    def input_size(self):
        #input_size = len(self.data[0]['target'][0])
        input_size = 1
        #if self.use_feats:
        #    # Multiplied by 2 because of sin and cos
        #    input_size += len(self.data[0]['feats'][0])
        for idx, (card, emb) in self.feats_info.items():
            if card != -1:
                input_size += emb
        return input_size

    @property
    def output_size(self):
        #output_size = len(self.data[0]['target'][0])
        output_size = 1
        return output_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #print(self.indices)
        ts_id = self.indices[idx][0]
        pos_id = self.indices[idx][1]

        el = self.enc_len
        dl = self.dec_len

        ex_input = self.data[ts_id]['target'][ pos_id : pos_id+el ]
        ex_target = self.data[ts_id]['target'][ pos_id+el : pos_id+el+dl ]
        #print('after', ex_input.shape, ex_target.shape, ts_id, pos_id)
        if self.tsid_map is None:
            mapped_id = ts_id
        else:
            mapped_id = self.tsid_map[ts_id]
        ex_input = self.input_norm.normalize(ex_input, mapped_id)#.unsqueeze(-1)
        ex_target = self.target_norm.normalize(ex_target, mapped_id)#.unsqueeze(-1)

        ex_input_feats = self.data[ts_id]['feats'][ pos_id : pos_id+el ]
        ex_target_feats = self.data[ts_id]['feats'][ pos_id+el : pos_id+el+dl ]
        ex_input_feats_norm = []
        ex_target_feats_norm = []
        for i in range(len(self.feats_info)):
            if self.feats_norms.get(i, -1) != -1:
                ex_input_feats_norm.append(self.feats_norms[i].normalize(
                    ex_input_feats[:, i], mapped_id)
                )
                ex_target_feats_norm.append(self.feats_norms[i].normalize(
                    ex_target_feats[:, i], mapped_id)
                )
            else:
                ex_input_feats_norm.append(ex_input_feats[:, i:i+1])
                ex_target_feats_norm.append(ex_target_feats[:, i:i+1])
        ex_input_feats = torch.cat(ex_input_feats_norm, dim=-1)
        ex_target_feats = torch.cat(ex_target_feats_norm, dim=-1)

        #i_res = self.enc_len - len(ex_input)
        #ex_input = torch.cat(
        #    [torch.zeros([i_res] + list(ex_input.shape[1:])), ex_input],
        #    dim=0
        #)
        #ex_input_feats = torch.cat(
        #    [torch.zeros([i_res] +list(ex_input_feats.shape[1:])), ex_input_feats],
        #    dim=0
        #)

        #print(ex_input.shape, ex_target.shape, ex_input_feats.shape, ex_target_feats.shape)

        return (
            ex_input, ex_target,
            ex_input_feats, ex_target_feats,
            mapped_id,
            torch.FloatTensor([ts_id, pos_id])
        )

    def collate_fn(self, batch):
        num_items = len(batch[0])
        batched = [[] for _ in range(len(batch[0]))]
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                batched[j].append(torch.tensor(batch[i][j]))

        batched_t = []
        for i, b in enumerate(batched):
            batched_t.append(torch.stack(b, dim=0))
            #print(i)
        #batched = [torch.stack(b, dim=0) for b in batched]

        return batched_t

class DataProcessor(object):
    """docstring for DataProcessor"""
    def __init__(self, args):
        super(DataProcessor, self).__init__()
        self.args = args

        if args.dataset_name in ['ett']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_ett(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['Solar']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_Solar(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['etthourly']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_etthourly(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['aggtest']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_aggtest(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['electricity']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_electricity(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['foodinflation']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_foodinflation(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)
        elif args.dataset_name in ['foodinflationmonthly']:
            (
                data_train, data_dev, data_test,
                dev_tsid_map, test_tsid_map,
                feats_info
            ) = parse_foodinflationmonthly(args.dataset_name, args.N_input, args.N_output, t2v_type=args.t2v_type)


        if args.use_feats:
            assert 'feats' in data_train[0].keys()

        self.data_train = data_train
        self.data_dev = data_dev
        self.data_test = data_test
        self.dev_tsid_map = dev_tsid_map
        self.test_tsid_map = test_tsid_map
        self.feats_info = feats_info


    def get_processed_data(self, args, agg_method, K):

        lazy_dataset_train = TimeSeriesDataset(
            self.data_train, args.N_input, args.N_output,
            which_split='train',
            norm_type=args.normalize,
            feats_info=self.feats_info,
        )
        print('Number of chunks in train data:', len(lazy_dataset_train))
        norm = lazy_dataset_train.input_norm
        dev_norm, test_norm = norm, norm
        feats_norms = lazy_dataset_train.feats_norms
        #for i in range(len(self.data_dev)):
        #   dev_norm.append(norm[self.dev_tsid_map[i]])
        #for i in range(len(self.data_test)):
        #   test_norm.append(norm[self.test_tsid_map[i]])
        #dev_norm, test_norm = np.stack(dev_norm), np.stack(test_norm)
        #import ipdb
        #ipdb.set_trace()
        lazy_dataset_dev = TimeSeriesDataset(
            self.data_dev, args.N_input, args.N_output,
            input_norm=dev_norm, which_split='dev',
            #target_norm=Normalizer(self.data_dev, 'same'),
            target_norm=dev_norm,
            feats_info=self.feats_info,
            tsid_map=self.dev_tsid_map,
            feats_norms=feats_norms,
            train_obj=lazy_dataset_train
        )
        print('Number of chunks in dev data:', len(lazy_dataset_dev))
        lazy_dataset_test = TimeSeriesDataset(
            self.data_test, args.N_input, args.N_output,
            input_norm=test_norm, which_split='test',
            #target_norm=test_norm,
            target_norm=Normalizer(self.data_test, 'same'),
            feats_info=self.feats_info,
            tsid_map=self.test_tsid_map,
            feats_norms=feats_norms,
            train_obj=lazy_dataset_train
        )
        print('Number of chunks in test data:', len(lazy_dataset_test))
        if len(lazy_dataset_train) >= args.batch_size:
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size
            while len(lazy_dataset_train) // batch_size < 10:
                batch_size = batch_size // 2
        #import ipdb ; ipdb.set_trace()
        if self.args.dataset_name in ['aggtest']:
            train_shuffle = False
        else:
            train_shuffle = True
        trainloader = DataLoader(
            lazy_dataset_train, batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=12, pin_memory=True,
            #collate_fn=lazy_dataset_train.collate_fn
        )
        devloader = DataLoader(
            lazy_dataset_dev, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=12, pin_memory=True,
            #collate_fn=lazy_dataset_dev.collate_fn
        )
        testloader = DataLoader(
            lazy_dataset_test, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=12, pin_memory=True,
            #collate_fn=lazy_dataset_test.collate_fn
        )
 
        return {
            'trainloader': trainloader,
            'devloader': devloader,
            'testloader': testloader,
            'N_input': lazy_dataset_test.enc_len,
            'N_output': lazy_dataset_test.dec_len,
            'input_size': lazy_dataset_test.input_size,
            'output_size': lazy_dataset_test.output_size,
            'train_norm': norm,
            'dev_norm': dev_norm,
            'test_norm': test_norm,
            'feats_info': self.feats_info,
            'dev_tsid_map': lazy_dataset_dev.tsid_map,
            'test_tsid_map': lazy_dataset_test.tsid_map
        }

