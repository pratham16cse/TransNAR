import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cvxpy as cp
import pywt
from scipy.linalg import block_diag

import utils

device = 'cuda'

class DILATE(torch.nn.Module):
    """docstring for DILATE"""
    def __init__(self, base_models_dict, device):
        super(DILATE, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device

    def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
        return self.base_models_dict[1].to(self.device)(feats_in_dict[1], inputs_dict[1], feats_tgt_dict[1])

class MSE(torch.nn.Module):
    """docstring for MSE"""
    def __init__(self, base_models_dict, device):
        super(MSE, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device

    def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
        return self.base_models_dict[1](feats_in_dict[1].to(self.device), inputs_dict[1].to(self.device), feats_tgt_dict[1].to(self.device))

class NLL(torch.nn.Module):
    """docstring for NLL"""
    def __init__(self, base_models_dict, device):
        super(NLL, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device

    def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
        return self.base_models_dict[1](feats_in_dict[1].to(self.device), inputs_dict[1].to(self.device), feats_tgt_dict[1].to(self.device))

class CNNRNN(torch.nn.Module):
    """docstring for NLL"""
    def __init__(self, base_models_dict, device):
        super(CNNRNN, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device

    def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
        return self.base_models_dict[1](
            feats_in_dict[1].to(self.device),
            inputs_dict[1].to(self.device),
            feats_tgt_dict[1].to(self.device)
        )

class RNNNLLNAR(torch.nn.Module):
    """docstring for NLL"""
    def __init__(self, base_models_dict, device, is_oracle=False, covariance=False):
        super(RNNNLLNAR, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device
        self.is_oracle = is_oracle
        self.covariance = covariance

    def forward(self, dataset, norms, which_split):
        feats_in = dataset['sum'][1][2].to(self.device)
        inputs = dataset['sum'][1][0].to(self.device)
        feats_tgt = dataset['sum'][1][3].to(self.device)
        #target = dataset['sum'][1][1].to(self.device)
        #if self.is_oracle:
        #    target = dataset['sum'][1][1].to(self.device)
        #else:
        #    target = None
        ids = dataset['sum'][1][4].cpu()

        mdl = self.base_models_dict['sum'][1]
        with torch.no_grad():
            out = mdl(feats_in, inputs, feats_tgt)
            if mdl.is_signature:
                if mdl.estimate_type in ['point']:
                    pred_mu, _, _ = out
                elif mdl.estimate_type in ['variance']:
                    pred_mu, pred_std, _, _ = out
                elif mdl.estimate_type in ['covariance']:
                    pred_mu, pred_std, pred_v, _, _ = out
                elif mdl.estimate_type in ['bivariate']:
                    pred_mu, pred_std, _, _, _ = out
            else:
                if mdl.estimate_type in ['point']:
                    pred_mu = out
                elif mdl.estimate_type in ['variance']:
                    pred_mu, pred_std = out
                elif mdl.estimate_type in ['covariance']:
                    pred_mu, pred_std, pred_v = out
                elif mdl.estimate_type in ['bivariate']:
                    pred_mu, pred_std, _ = out
        pred_mu = pred_mu.cpu()

        if mdl.estimate_type in ['covariance']:
            pred_std = pred_std.cpu()
            pred_d = pred_std**2
            pred_v = pred_v.cpu()

            dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                torch.squeeze(pred_mu, dim=-1), pred_v, torch.squeeze(pred_d, dim=-1)
            )
            pred_std = torch.sqrt(
                torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1).unsqueeze(dim=-1)
            )
            if which_split in ['test']:
                raise NotImplementedError
                pred_std = norms['sum'][1].unnormalize(pred_std[..., 0], ids=ids, is_var=True).unsqueeze(-1)
        elif mdl.estimate_type in ['variance', 'bivariate']:
            pred_std = pred_std.cpu()
            pred_d = pred_std**2
            pred_v = torch.ones_like(pred_mu) * 1e-9
            if which_split in ['test']:
                pred_std = torch.sqrt(
                    norms['sum'][1].unnormalize(pred_d[..., 0], ids=ids, is_var=True).unsqueeze(-1)
                )
                pred_d = norms['sum'][1].unnormalize(pred_d[..., 0], ids=ids, is_var=True).unsqueeze(-1)
        else:
            pred_d = torch.ones_like(pred_mu) * 1e-9
            pred_v = torch.ones_like(pred_mu) * 1e-9
            pred_std = torch.ones_like(pred_mu) * 1e-9

        if which_split in ['test']:
            pred_mu = norms['sum'][1].unnormalize(pred_mu[..., 0], ids=ids, is_var=False).unsqueeze(-1)

        #import ipdb ; ipdb.set_trace()

        return (pred_mu, pred_d, pred_v, pred_std)

class RNN_MSE_NAR(torch.nn.Module):
    
    def __init__(self, base_models_dict, device):
        super(RNN_MSE_NAR, self).__init__()
        self.base_models_dict = base_models_dict
        self.device = device

    def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
        return self.base_models_dict[1](
            feats_in_dict[1].to(self.device),
            inputs_dict[1].to(self.device),
            feats_tgt_dict[1].to(self.device)
        )

class KLInference(torch.nn.Module):
    """docstring for DualTPP"""
    def __init__(self, K_list, base_models_dict, aggregates, device, opt_normspace=False):
        '''
        K: int
            number of steps to aggregate at each level
        base_models_dict: dict
            key: level in the hierarchy
            value: base model at the level 'key'
        '''
        super(KLInference, self).__init__()
        self.K_list = K_list
        self.base_models_dict = base_models_dict
        self.aggregates = aggregates
        self.device = device
        self.opt_normspace = opt_normspace

    def aggregate_data(self, y, agg, K, is_var):
        if agg == 'sum' and not is_var:
            return 1./y.shape[1] * cp.sum(y, axis=1, keepdims=True)
        elif agg == 'sum' and is_var:
            return 1./y.shape[1]**2 * cp.sum(y, axis=1, keepdims=True)
        elif agg == 'slope':
            if K==1:
                return y
            #x = torch.arange(y.shape[0], dtype=torch.float)
            x = torch.arange(y.shape[1], dtype=torch.float).unsqueeze(0).repeat(y.shape[0], 1)
            m_x = x.mean(dim=1, keepdims=True)
            s_xx = ((x-m_x)**2).sum(dim=1, keepdims=True)
            a = (x - m_x) / s_xx
            if not is_var:
                w = cp.sum(cp.multiply(a, y), axis=1, keepdims=True)
                return w
            else:
                w = cp.sum(cp.multiply(a**2, y), axis=1, keepdims=True)
                return w

    def KL_loss(self, x_mu, x_var, mu, std):
        return cp.sum(cp.log(std) - cp.log(x_var)/2. + (x_var + (mu-x_mu)**2)/(2*std**2) - 0.5)
        #return cp.sum(cp.log(std) + (x_var + (mu-x_mu)**2)/(2*std**2) - 0.5)

    def log_prob(self, x_, means, std):
        return -cp.sum(cp.log(1/(((2*np.pi)**0.5)*std)) - (((x_ - means)**2) / (2.*(std)**2)))

    #def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
    def forward(self, dataset, norms, which_split):

        #inputs_dict = dataset['sum'][1]
        #import ipdb ; ipdb.set_trace()

        #norm_dict_np = dict()
        #for lvl in norm_dict.keys():
        #    norm_dict_np[lvl] = norm_dict[lvl].detach().numpy()

        params_dict = {}
        ids_dict = {}
        for agg in self.aggregates:
            params_dict[agg] = {}
            ids_dict[agg] = {}
            for level in self.K_list:
                model = self.base_models_dict[agg][level]
                inputs = dataset[agg][level][0]
                feats_in, feats_tgt = dataset[agg][level][2], dataset[agg][level][3]
                ids = dataset[agg][level][4].cpu()

                with torch.no_grad():
                    if model.estimate_type in ['point']:
                        means, d, v = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['variance']:
                        means, d = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['covariance']:
                        means, d, v = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['bivariate']:
                        means, d, _ = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )

                means = means.cpu()
                if model.estimate_type is 'covariance':
                    d = d.cpu()
                    v = v.cpu()

                    dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                        means.squeeze(dim=-1), v, d.squeeze(dim=-1)
                    )
                    stds = torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1).unsqueeze(dim=-1)
                elif model.estimate_type in ['variance', 'bivariate']:
                    stds = d.cpu()
                    v = torch.ones_like(means) * 1e-9
                    if not self.opt_normspace:
                        stds = norms[agg][level].unnormalize(stds[..., 0], ids=ids, is_var=True).unsqueeze(-1)
                else:
                    d = torch.ones_like(means) * 1e-9
                    v = torch.ones_like(means) * 1e-9
                    stds = torch.ones_like(means) * 1e-9

                if not self.opt_normspace:
                    means = norms[agg][level].unnormalize(means[..., 0], ids=ids, is_var=False).unsqueeze(-1)

                params = [means, stds, d, v]
                params_dict[agg][level] = params
                ids_dict[agg][level] = ids

        #import ipdb ; ipdb.set_trace()

        base_lvl = self.aggregates[0]
        bs, N = params_dict[base_lvl][1][0].shape[0], params_dict[base_lvl][1][0].shape[1]
        x_mu = cp.Variable((bs, N))
        x_var = cp.Variable((bs, N))
        x_mu_dict, x_var_dict = {}, {}
        opt_loss = 0.
        all_preds_mu, all_preds_std = [], []

        opt_bs = bs
        for bch in range(0, bs, opt_bs):
            #print('Example:', bch)
            try:
                for agg in self.aggregates:
                    x_mu_dict[agg], x_var_dict[agg] = {}, {}
                    for lvl in self.K_list:

                        base_lvl_present = False
                        if lvl==1: # If lvl=1 present in other aggregates, ignore it
                            #import ipdb ; ipdb.set_trace()
                            other_aggs = set(self.aggregates) - {agg}
                            for other_agg in other_aggs:
                                if x_mu_dict.get(other_agg, -1) is not -1:
                                    base_lvl_present = True

                        if not base_lvl_present:
                            #x_mu_dict[agg][lvl], x_var_dict[agg][lvl] = {}, {}
                            lvl_x_mu, lvl_x_var = [], []
                            for i in range(0, N, lvl):
                                lvl_x_mu.append(self.aggregate_data(x_mu[bch:bch+opt_bs, i:i+lvl], agg, lvl, is_var=False))
                                lvl_x_var.append(self.aggregate_data(x_var[bch:bch+opt_bs, i:i+lvl], agg, lvl, is_var=True))
                            x_mu_dict[agg][lvl] = lvl_x_mu
                            x_var_dict[agg][lvl] = lvl_x_var

                        #for lvl, params in params_dict[agg].items():
                            params = params_dict[agg][lvl]
                            for idx, _ in enumerate(range(0, N, lvl)):
                                #import ipdb ; ipdb.set_trace()
                                loss = self.KL_loss(
                                    x_mu_dict[agg][lvl][idx],
                                    x_var_dict[agg][lvl][idx],
                                    params[0][bch:bch+opt_bs, idx:idx+1, 0].detach(),
                                    params[1][bch:bch+opt_bs, idx:idx+1, 0].detach()
                                )
                                opt_loss += loss
                #import ipdb ; ipdb.set_trace()

                objective = cp.Minimize(opt_loss)

                constraints = [x_var>=1e-6]

                prob = cp.Problem(objective, constraints)

                #x_mu.value = params_dict[base_lvl][1][0]
                #x_var.value = params_dict[base_lvl][1][1]

                opt_loss = prob.solve()
            except cp.error.SolverError:
                for agg in self.aggregates:
                    x_mu_dict[agg], x_var_dict[agg] = {}, {}
                    for lvl in self.K_list:

                        base_lvl_present = False
                        if lvl==1: # If lvl=1 present in other aggregates, ignore it
                            #import ipdb ; ipdb.set_trace()
                            other_aggs = set(self.aggregates) - {agg}
                            for other_agg in other_aggs:
                                if x_mu_dict.get(other_agg, -1) is not -1:
                                    base_lvl_present = True

                        if not base_lvl_present:
                            #x_mu_dict[agg][lvl], x_var_dict[agg][lvl] = {}, {}
                            lvl_x_mu, lvl_x_var = [], []
                            for i in range(0, N, lvl):
                                lvl_x_mu.append(self.aggregate_data(x_mu[bch:bch+opt_bs, i:i+lvl], agg, lvl, is_var=False))
                                lvl_x_var.append(self.aggregate_data(x_var[bch:bch+opt_bs, i:i+lvl], agg, lvl, is_var=True))
                            x_mu_dict[agg][lvl] = lvl_x_mu
                            x_var_dict[agg][lvl] = lvl_x_var

                        #for lvl, params in params_dict[agg].items():
                            params = params_dict[agg][lvl]
                            for idx, _ in enumerate(range(0, N, lvl)):
                                #import ipdb ; ipdb.set_trace()
                                loss = self.log_prob(
                                    x_mu_dict[agg][lvl][idx],
                                    params[0][bch:bch+opt_bs, idx:idx+1, 0].detach(),
                                    params[1][bch:bch+opt_bs, idx:idx+1, 0].detach()
                                )
                                opt_loss += loss
                #import ipdb ; ipdb.set_trace()

                objective = cp.Minimize(opt_loss)

                #constraints = [x_var>=1e-9]

                prob = cp.Problem(objective)#, constraints)

                #opt_loss = prob.solve(solver='SCS')
                opt_loss = prob.solve()

                x_var.value = params_dict[base_lvl][1][1].detach().numpy()[..., 0]**2

            #import ipdb ; ipdb.set_trace()

            all_preds_mu.append(torch.FloatTensor(x_mu.value).unsqueeze(dim=-1))
            all_preds_std.append(torch.sqrt(torch.FloatTensor(x_var.value).unsqueeze(dim=-1)))

        #all_preds_mu = torch.FloatTensor(x_mu.value).unsqueeze(dim=-1)
        #all_preds_std = torch.sqrt(torch.FloatTensor(x_var.value).unsqueeze(dim=-1))
        all_preds_mu = torch.cat(all_preds_mu, dim=0)
        all_preds_std = torch.cat(all_preds_std, dim=0)
        if which_split in ['test'] and self.opt_normspace:
            all_preds_mu = norms[base_lvl][1].unnormalize(
                all_preds_mu[..., 0], ids=ids_dict[base_lvl][1], is_var=False
            ).unsqueeze(-1)
            all_preds_std = norms[base_lvl][1].unnormalize(
                all_preds_std[..., 0], ids=ids_dict[base_lvl][1], is_var=True
            ).unsqueeze(-1)
        if which_split in ['dev'] and not self.opt_normspace:
            all_preds_mu = norms[base_lvl][1].normalize(
                all_preds_mu[..., 0], ids=ids_dict[base_lvl][1], is_var=False
            )
            all_preds_std = norms[base_lvl][1].normalize(
                all_preds_std[..., 0], ids=ids_dict[base_lvl][1], is_var=True
            )

        d = params_dict[base_lvl][1][2]
        v = params_dict[base_lvl][1][3]

        return all_preds_mu, d, v, all_preds_std

class KLInferenceSGD(torch.nn.Module):
    """docstring for DualTPP"""
    def __init__(
        self, K_list, base_models_dict, aggregates, lr, device,
        solve_mean, solve_std, opt_normspace=False, kldirection='qp',
        covariance=False
    ):
        '''
        K: int
            number of steps to aggregate at each level
        base_models_dict: dict
            key: level in the hierarchy
            value: base model at the level 'key'
        '''
        super(KLInferenceSGD, self).__init__()
        self.K_list = K_list
        self.base_models_dict = base_models_dict
        self.aggregates = aggregates
        self.device = device
        self.solve_mean = solve_mean
        self.solve_std = solve_std
        self.opt_normspace = opt_normspace
        self.kldirection = kldirection
        self.covariance = covariance
        self.lr = lr

    def get_a(self, agg_type, K):
        if agg_type in ['sum']:
            a = 1./K * torch.ones(K)
        elif agg_type in ['slope']:
            x = torch.arange(K, dtype=torch.float)
            m_x = x.mean()
            s_xx = ((x-m_x)**2).sum()
            a = (x - m_x) / s_xx
        return a

    def aggregate_data(self, y, v, agg, K, a, is_var):
        a = a.unsqueeze(0).repeat(y.shape[0], 1)
        if K==1:
            return y
        if is_var==False:
            y_a = (a*y).sum(dim=1, keepdims=True)
        else:
            w_d = (a**2*y).sum(dim=1, keepdims=True)
            if self.covariance:
                w_v = (((a.unsqueeze(-1)*v).sum(-1)**2)).sum(dim=1, keepdims=True)
                y_a = w_d + w_v
            else:
                y_a = w_d
        return y_a

    def aggregate_data_bak(self, y, agg, K, is_var):
        if agg == 'sum' and not is_var:
            return 1./y.shape[1] * torch.sum(y, axis=1, keepdims=True)
        elif agg == 'sum' and is_var:
            return 1./y.shape[1]**2 * torch.sum(y, axis=1, keepdims=True)
        elif agg == 'slope':
            if K==1:
                return y
            #x = torch.arange(y.shape[0], dtype=torch.float)
            x = torch.arange(y.shape[1], dtype=torch.float).unsqueeze(0).repeat(y.shape[0], 1)
            m_x = x.mean(dim=1, keepdims=True)
            s_xx = ((x-m_x)**2).sum(dim=1, keepdims=True)
            a = (x - m_x) / s_xx
            if not is_var:
                w = torch.sum(a*y, axis=1, keepdims=True)
                return w
            else:
                w = torch.sum(a**2*y, axis=1, keepdims=True)
                return w

    #def log_prob(self, x_, means, std):
    #    return -cp.sum(cp.log(1/(((2*np.pi)**0.5)*std)) - (((x_ - means)**2) / (2.*(std)**2)))

    def get_params_dict(self, dataset, norms):

        params_dict = {}
        ids_dict = {}
        for agg in self.aggregates:
            params_dict[agg] = {}
            ids_dict[agg] = {}
            for level in self.K_list:
                model = self.base_models_dict[agg][level]
                inputs = dataset[agg][level][0]
                feats_in, feats_tgt = dataset[agg][level][2], dataset[agg][level][3]
                ids = dataset[agg][level][4].cpu()

                with torch.no_grad():
                    if model.estimate_type in ['point']:
                        means = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['variance']:
                        means, stds = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['covariance']:
                        means, stds, v = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )
                    elif model.estimate_type in ['bivariate']:
                        means, stds, _ = model(
                            feats_in.to(self.device), inputs.to(self.device),
                            feats_tgt.to(self.device)
                        )

                means = means.cpu()
                if model.estimate_type is 'covariance':
                    stds = stds.cpu()
                    d = stds**2
                    v = v.cpu()

                    dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                        means.squeeze(dim=-1), v, d.squeeze(dim=-1)
                    )
                    stds = torch.sqrt(
                        torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1).unsqueeze(dim=-1)
                    )
                elif model.estimate_type in ['variance', 'bivariate']:
                    stds = stds.cpu()
                    d = stds**2
                    v = torch.ones_like(means) * 1e-9
                    if not self.opt_normspace:
                        stds = torch.sqrt(
                            norms[agg][level].unnormalize(d[..., 0], ids=ids, is_var=True).unsqueeze(-1)
                        )
                        d = norms['sum'][1].unnormalize(d[..., 0], ids=ids, is_var=True).unsqueeze(-1)
                else:
                    d = torch.ones_like(means) * 1e-9
                    v = torch.ones_like(means) * 1e-9
                    stds = torch.ones_like(means) * 1e-9

                if not self.opt_normspace:
                    means = norms[agg][level].unnormalize(means[..., 0], ids=ids, is_var=False).unsqueeze(-1)

                params = [means, stds, d, v]
                params_dict[agg][level] = params
                ids_dict[agg][level] = ids

        return params_dict, ids_dict

    def get_A(self, agg, K, bs, N, sigma):
        #A_ = torch.block_diag(*[torch.ones(K)*1./K]*(N//K)).unsqueeze(dim=0).repeat(bs, 1, 1)
        if agg == 'sum':
            a = torch.ones(K)*1./K
        elif agg == 'slope':
            if K==1:
                a = torch.ones(K)
            else:
                x = torch.arange(K, dtype=torch.float)
                m_x = x.mean()
                s_xx = ((x-m_x)**2).sum()
                a = (x - m_x) / s_xx
        elif agg == 'haar':
            a_ = torch.ones(K)*1./K
            s = K // 2
            a = torch.cat([a_[:s]*-1., a_[s:]], dim=0)

        #import ipdb ; ipdb.set_trace()

        A_ = torch.block_diag(*[torch.ones(K)*a]*(N//K)).unsqueeze(dim=0).repeat(bs, 1, 1)
        sig_ = torch.block_diag(*[torch.ones(K)]*(N//K)).unsqueeze(dim=0).repeat(bs, 1, 1)
        #if K>1:
        #    import ipdb ; ipdb.set_trace()
        sig_ = sig_ * 1./sigma.repeat(1,1,sig_.shape[-1])
        #import ipdb ; ipdb.set_trace()

        A_ = A_ * sig_

        return A_

    def solve_base_level_mean(self, params_dict, bs, N):
        max_K = max(self.K_list)
        x = []
        for i in range(0, N, max_K):
            A = []
            flg1 = False
            for agg in self.aggregates:
                for K in self.K_list:
                    idx_1 = i//K
                    idx_2 = idx_1 + max_K//K
                    #print(agg, K, i, i+max_K//K, params_dict[agg][K][1].shape, params_dict[agg][K][1][..., i:i+max_K//K, :].shape)
                    if K==1 and flg1 == False: # If K=1 present
                        A_ = self.get_A(agg, K, bs, max_K, params_dict[agg][K][1][..., idx_1:idx_2, :])
                        A.append(A_)
                        flg1 = True
                    elif K>1:
                        A_ = self.get_A(agg, K, bs, max_K, params_dict[agg][K][1][..., idx_1:idx_2, :])
                        #import ipdb ; ipdb.set_trace()
                        A.append(A_)
                    #import ipdb ; ipdb.set_trace()
            A = torch.cat(A, dim=1)

            b = []
            flg2 = False
            for agg in self.aggregates:
                for K in self.K_list:
                    idx_1 = i//K
                    idx_2 = idx_1 + max_K//K
                    if K==1 and flg2==False:
                        b_ = params_dict[agg][K][0][..., idx_1:idx_2, :] / params_dict[agg][K][1][..., idx_1:idx_2, :]
                        b.append(b_)
                        flg2 = True
                    elif K>1:
                        b_ = params_dict[agg][K][0][..., idx_1:idx_2, :] / params_dict[agg][K][1][..., idx_1:idx_2, :]
                        b.append(b_)

                    #import ipdb ; ipdb.set_trace()
            b = torch.cat(b, dim=1)

            x_ = torch.matmul(
                torch.inverse(torch.matmul(A.transpose(1, 2), A)), torch.matmul(A.transpose(1, 2), b)
            )
            x.append(x_)
        #import ipdb ; ipdb.set_trace()
        x = torch.cat(x, dim=1)

        return x

    def initialize_params(self, base_mu, base_sigma):
        #import ipdb ; ipdb.set_trace()
        self.x_mu = torch.nn.Parameter(torch.clone(base_mu).squeeze(-1))
        self.x_d = torch.nn.Parameter(torch.clone(base_sigma).squeeze(-1)**2)
        x_v = torch.ones((base_mu.shape[0], base_mu.shape[1], 4), dtype=torch.float) * 1e-2
        if self.covariance:
            self.x_v = torch.nn.Parameter(x_v)
        else:
            self.x_v = x_v

    def x_dc(self):
        return self.x_d.clamp(min=1e-4)

    def x_var(self):
        return self.x_dc() + (self.x_v**2).sum(dim=-1)

    def KL_loss(self, x_mu, x_var, mu, std):
        #import ipdb ; ipdb.set_trace()
        #return cp.sum(cp.log(std) + (x_var + (mu-x_mu)**2)/(2*std**2) - 0.5)
        #kl = torch.log(std) - torch.log(x_var)/2. + (x_var + (mu-x_mu)**2)/(2*std**2) - 0.5
        #return torch.sum(torch.log(kl + 1e-4))
        if self.kldirection == 'qp':
            return torch.mean(torch.log(std) - torch.log(x_var)/2. + (x_var)/(2*std**2) - 0.5)
        elif self.kldirection == 'pq':
            return torch.mean(torch.log(x_var)/2. - torch.log(std) + (std**2)/(2*x_var) - 0.5)

    #def forward(self, feats_in_dict, inputs_dict, feats_tgt_dict, norm_dict, targets_dict=None):
    def forward(self, dataset, norms, which_split):

        #import ipdb ; ipdb.set_trace()

        params_dict, ids_dict = self.get_params_dict(dataset, norms)

        #import ipdb ; ipdb.set_trace()

        base_lvl = self.aggregates[0]
        bs, N = params_dict[base_lvl][1][0].shape[0], params_dict[base_lvl][1][0].shape[1]

        # Solve for base level mean
        if self.solve_mean:
            base_lvl_mu = self.solve_base_level_mean(params_dict, bs, N)
        else:
            base_lvl_mu = params_dict[base_lvl][1][0]

        base_lvl_std = params_dict[base_lvl][1][1]

        #self.initialize_params(params_dict[base_lvl][1][0], params_dict[base_lvl][1][1])
        self.initialize_params(base_lvl_mu, base_lvl_std)

        all_preds_mu, all_preds_std = [], []
        all_preds_d, all_preds_v = [], []

        #if self.covariance: lr = 0.001
        #else: lr = 0.001
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        a_dict = {}
        for agg in self.aggregates:
            a_dict[agg] = {}
            for K in self.K_list:
                a_dict[agg][K] = utils.get_a(agg, K)

        if self.solve_std:
            opt_bs = bs
            # Minimize the KL_loss
            for bch in range(0, bs, opt_bs):
                #for s in range(4000):
                s, opt_loss_prev, opt_grad_prev = 0, 10000000., 10000000.
                while True:
                    x_mu_dict, x_var_dict = {}, {}
                    opt_loss = 0.
                    #print('Example:', bch)
                    for agg in self.aggregates:
                        x_mu_dict[agg], x_var_dict[agg] = {}, {}
                        for lvl in self.K_list:

                            params = params_dict[agg][lvl]

                            base_lvl_present = False
                            if lvl==1: # If lvl=1 present in other aggregates, ignore it
                                #import ipdb ; ipdb.set_trace()
                                other_aggs = set(self.aggregates) - {agg}
                                for other_agg in other_aggs:
                                    if x_mu_dict.get(other_agg, -1) is not -1:
                                        base_lvl_present = True

                            if not base_lvl_present and lvl==1:
                                    opt_loss += self.KL_loss(
                                        self.x_mu[bch:bch+opt_bs],
                                        self.x_var()[bch:bch+opt_bs],
                                        params[0][bch:bch+opt_bs, ..., 0].detach(),
                                        params[1][bch:bch+opt_bs, ..., 0].detach()
                                    )

                            lvl_x_mu, lvl_x_var = [], []
                            if lvl != 1:
                                for i in range(0, N, lvl):
                                    lvl_x_mu.append(
                                        utils.aggregate_window(
                                            self.x_mu[bch:bch+opt_bs, i:i+lvl],
                                            a_dict[agg][lvl], False,
                                        )
                                    )
                                    if self.covariance:
                                        v_for_agg = self.x_v[bch:bch+opt_bs, i:i+lvl]
                                    else:
                                        v_for_agg = None
                                    lvl_x_var.append(
                                        utils.aggregate_window(
                                            self.x_dc()[bch:bch+opt_bs, i:i+lvl],
                                            a_dict[agg][lvl], True, 
                                            v_for_agg,
                                        )
                                    )
                                x_mu_dict[agg][lvl] = lvl_x_mu
                                x_var_dict[agg][lvl] = lvl_x_var

                                #import ipdb ; ipdb.set_trace()
                                for idx, _ in enumerate(range(0, N, lvl)):
                                    opt_loss += self.KL_loss(
                                        x_mu_dict[agg][lvl][idx],
                                        x_var_dict[agg][lvl][idx],
                                        params[0][bch:bch+opt_bs, idx:idx+1, 0].detach(),
                                        params[1][bch:bch+opt_bs, idx:idx+1, 0].detach()
                                    )
                    #import ipdb ; ipdb.set_trace() 
                    optimizer.zero_grad()
                    opt_loss.backward()
                    optimizer.step()
                    s += 1

                    if s % 100 == 0:
                    #if True:
                        if self.covariance:
                            print('opt_loss:', opt_loss, 'grad:', self.x_d.grad.norm(), self.x_v.grad.norm(), self.x_d.grad.norm()+self.x_v.grad.norm())
                        else:
                            print('opt_loss:', opt_loss, 'grad:', self.x_d.grad.norm())
                    #import ipdb ; ipdb.set_trace()

                    #if (torch.abs(opt_loss_prev - opt_loss)/opt_loss_prev).item() <= 1e-3:
                    #if (self.x_d.grad.norm()) <= 1e-2:
                    if self.covariance:
                        opt_grad = self.x_d.grad.norm()+self.x_v.grad.norm()
                    else:
                        opt_grad = self.x_d.grad.norm()
                    condition = (
                        opt_grad<=1e-2 \
                        #or (torch.abs(opt_loss_prev - opt_loss)/opt_loss_prev).item() <= 1e-2 \
                        #or (torch.abs(opt_grad_prev - opt_grad)).item() <= 1e-2
                        or (s>=10000)
                    )
                    #condition = opt_grad<=1e-2 or (torch.abs(opt_loss_prev - opt_loss)/opt_loss_prev).item() <= 1e-2
                    if condition:
                        print('Stopping after {} steps'.format(s))
                        break
                    else:
                        opt_loss_prev = opt_loss
                        opt_grad_prev = opt_grad

                #import ipdb ; ipdb.set_trace()
                all_preds_mu.append(self.x_mu.unsqueeze(dim=-1))
                #all_preds_std.append(torch.sqrt(self.x_d.unsqueeze(dim=-1)))
                all_preds_std.append(torch.sqrt(self.x_var().unsqueeze(dim=-1)))
                all_preds_d.append(self.x_dc().unsqueeze(dim=-1))
                all_preds_v.append(self.x_v)

        #all_preds_mu = torch.FloatTensor(x_mu.value).unsqueeze(dim=-1)
        #all_preds_std = torch.sqrt(torch.FloatTensor(x_var.value).unsqueeze(dim=-1))
        all_preds_mu = base_lvl_mu
        if self.solve_std:
            all_preds_std = torch.cat(all_preds_std, dim=0)
            all_preds_d = torch.cat(all_preds_d, dim=0)
            all_preds_v = torch.cat(all_preds_v, dim=0)
        else:
            all_preds_std = base_lvl_std
            all_preds_d = params_dict[base_lvl][1][2]
            all_preds_v = params_dict[base_lvl][1][3]
        #if which_split in ['test'] and self.opt_normspace:
        #    all_preds_mu = norms[base_lvl][1].unnormalize(
        #        all_preds_mu[..., 0], ids=ids_dict[base_lvl][1], is_var=False
        #    ).unsqueeze(-1)
        #    all_preds_std = norms[base_lvl][1].unnormalize(
        #        all_preds_std[..., 0], ids=ids_dict[base_lvl][1], is_var=True
        #    ).unsqueeze(-1)
        #if which_split in ['dev'] and not self.opt_normspace:
        #    all_preds_mu = norms[base_lvl][1].normalize(
        #        all_preds_mu[..., 0], ids=ids_dict[base_lvl][1], is_var=False
        #    )
        #    all_preds_std = norms[base_lvl][1].normalize(
        #        all_preds_std[..., 0], ids=ids_dict[base_lvl][1], is_var=True
        #    )

        #if self.covariance:
        #    import ipdb ; ipdb.set_trace()


        #return all_preds_mu, d, v, all_preds_std
        return all_preds_mu, all_preds_d, all_preds_v, all_preds_std
