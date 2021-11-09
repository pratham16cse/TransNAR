import sys
import os
import argparse
import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.base_models import get_base_model
from loss.dilate_loss import dilate_loss
from train import train_model, get_optimizer
from eval import eval_base_model, eval_inf_model, eval_aggregates
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import json
from torch.utils.tensorboard import SummaryWriter
import shutil
import properscoring as ps
import scipy.stats
import itertools

from functools import partial

from models import inf_models
import utils

os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "1000000"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('dataset_name', type=str, help='dataset_name')
#parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--N_input', type=int, default=-1,
                    help='number of input steps')
parser.add_argument('--N_output', type=int, default=-1,
                    help='number of output steps')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs', default=None)
parser.add_argument('--saved_models_dir', type=str,
                    help='Path to store all saved models', default=None)

parser.add_argument('--ignore_ckpt', action='store_true', default=False,
                    help='Start the training without loading the checkpoint')

parser.add_argument('--normalize', type=str, default=None,
                    choices=[
                        'same', 'zscore_per_series', 'gaussian_copula', 'log', 'zeroshift_per_series'
                    ],
                    help='Normalization type (avg, avg_per_series, quantile90, std)')
parser.add_argument('--epochs', type=int, default=-1,
                    help='number of training epochs')

parser.add_argument('--print_every', type=int, default=50,
                    help='Print test output after every print_every epochs')

parser.add_argument('--learning_rate', type=float, default=-1.,# nargs='+',
                   help='Learning rate for the training algorithm')
parser.add_argument('--hidden_size', type=int, default=-1,# nargs='+',
                   help='Number of units in the encoder/decoder state of the model')
parser.add_argument('--num_grulstm_layers', type=int, default=-1,# nargs='+',
                   help='Number of layers of the model')

parser.add_argument('--fc_units', type=int, default=16, #nargs='+',
                   help='Number of fully connected units on top of the encoder/decoder state of the model')

parser.add_argument('--batch_size', type=int, default=-1,
                    help='Input batch size')

parser.add_argument('--gamma', type=float, default=0.01, nargs='+',
                   help='gamma parameter of DILATE loss')
parser.add_argument('--alpha', type=float, default=0.5,
                   help='alpha parameter of DILATE loss')
parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                   help='Probability of applying teacher forcing to a batch')
parser.add_argument('--deep_std', action='store_true', default=False,
                    help='Extra layers for prediction of standard deviation')
parser.add_argument('--train_twostage', action='store_true', default=False,
                    help='Train base model in two stages -- train only \
                          mean in first stage, train both in second stage')
parser.add_argument('--mse_loss_with_nll', action='store_true', default=False,
                    help='Add extra mse_loss when training with nll')
parser.add_argument('--second_moment', action='store_true', default=False,
                    help='compute std as std = second_moment - mean')
parser.add_argument('--variance_rnn', action='store_true', default=False,
                    help='Use second RNN to compute variance or variance related values')
parser.add_argument('--input_dropout', type=float, default=0.0,
                    help='Dropout on input layer')

parser.add_argument('--v_dim', type=int, default=-1,
                   help='Dimension of V vector in LowRankGaussian')
parser.add_argument('--b', type=int, default=-1,
                   help='Number of correlation terms to sample for loss computation during training')

#parser.add_argument('--use_feats', action='store_true', default=False,
#                    help='Use time features derived from calendar-date and other covariates')
parser.add_argument('--use_feats', type=int, default=-1,
                    help='Use time features derived from calendar-date and other covariates')

parser.add_argument('--t2v_type', type=str,
                    choices=['local', 'idx', 'mdh_lincomb', 'mdh_parti'],
                    help='time2vec type', default=None)

parser.add_argument('--use_coeffs', action='store_true', default=False,
                    help='Use coefficients obtained by decomposition, wavelet, etc..')


# Hierarchical model arguments
parser.add_argument('--L', type=int, default=2,
                    help='number of levels in the hierarchy, leaves inclusive')

parser.add_argument('--K_list', type=int, nargs='*', default=[1],
                    help='List of bin sizes of each aggregation')

parser.add_argument('--wavelet_levels', type=int, default=2,
                    help='number of levels of wavelet coefficients')
parser.add_argument('--fully_connected_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a feed-forward network')
parser.add_argument('--transformer_agg_model', action='store_true', default=False,
                    help='If True, aggregate model will be a Transformer')
parser.add_argument('--plot_anecdotes', action='store_true', default=False,
                    help='Plot the comparison of various methods')
parser.add_argument('--save_agg_preds', action='store_true', default=False,
                    help='Save inputs, targets, and predictions of aggregate base models')

parser.add_argument('--device', type=str,
                    help='Device to run on', default=None)

# parameters for ablation study
parser.add_argument('--leak_agg_targets', action='store_true', default=False,
                    help='If True, aggregate targets are leaked to inference models')
parser.add_argument('--patience', type=int, default=20,
                    help='Stop the training if no improvement shown for these many \
                          consecutive steps.')
#parser.add_argument('--seed', type=int,
#                    help='Seed for parameter initialization',
#                    default=42)

# Parameters for ARTransformerModel
parser.add_argument('--kernel_size', type=int, default=-1,
                    help='Kernel Size of Conv (in ARTransformerModel)')
parser.add_argument('--nkernel', type=int, default=-1,
                    help='Number of kernels of Conv (in ARTransformerModel)')

parser.add_argument('--dim_ff', type=int, default=512,
                    help='Dimension of Feedforward (in ARTransformerModel)')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads (in ARTransformerModel)')


# Cross-validation parameters
parser.add_argument('--cv_inf', type=int, default=-1,
                    help='Cross-validate the Inference models based on score on dev data')

# Learning rate for Inference Model
parser.add_argument('--lr_inf', type=float, default=-1.,
                    help='Learning rate for SGD-based inference model')



args = parser.parse_args()

#args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.base_model_names = [
#    'rnn-q-nar',
#    'rnn-q-ar',
#    'trans-mse-nar',
#    'trans-q-nar',
#    'rnn-mse-ar',
#    'rnn-nll-ar',
#    'trans-mse-ar',
    'trans-nll-ar',
#    'gpt-nll-ar',
#    'gpt-mse-ar',
#    'gpt-nll-nar',
#    'gpt-mse-nar',
#    'informer-mse-nar',
#    'trans-bvnll-ar',
#    'trans-nll-atr',
#    'trans-fnll-ar',
#    'rnn-mse-nar',
#    'rnn-nll-nar',
#    'rnn-fnll-nar',
#    'oracle',
#    'oracleforecast'
#    'transsig-nll-nar',
]
args.aggregate_methods = [
    'sum',
    'slope',
]

args.inference_model_names = []
if 'rnn-q-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-NAR')
if 'rnn-mse-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-AR')
if 'rnn-q-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-Q-AR')
if 'trans-mse-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-NAR')
if 'trans-q-nar' in args.base_model_names:
    args.inference_model_names.append('TRANS-Q-NAR')
if 'rnn-mse-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-MSE-NAR')
if 'rnn-nll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-NAR')
if 'rnn-nll-ar' in args.base_model_names:
    args.inference_model_names.append('RNN-NLL-AR')
if 'trans-mse-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-MSE-AR')
if 'trans-nll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-AR')
if 'gpt-nll-ar' in args.base_model_names:
    args.inference_model_names.append('GPT-NLL-AR')
if 'gpt-mse-ar' in args.base_model_names:
    args.inference_model_names.append('GPT-MSE-AR')
if 'gpt-nll-nar' in args.base_model_names:
    args.inference_model_names.append('GPT-NLL-NAR')
if 'gpt-mse-nar' in args.base_model_names:
    args.inference_model_names.append('GPT-MSE-NAR')
if 'informer-mse-nar' in args.base_model_names:
    args.inference_model_names.append('INFORMER-MSE-NAR')
if 'trans-bvnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-BVNLL-AR')
if 'trans-nll-atr' in args.base_model_names:
    args.inference_model_names.append('TRANS-NLL-ATR')
if 'trans-fnll-ar' in args.base_model_names:
    args.inference_model_names.append('TRANS-FNLL-AR')
if 'rnn-fnll-nar' in args.base_model_names:
    args.inference_model_names.append('RNN-FNLL-NAR')
if 'oracle' in args.base_model_names:
    args.inference_model_names.append('oracle')
if 'oracleforecast' in args.base_model_names:
    args.inference_model_names.append('SimRetrieval')
if 'transsig-nll-nar' in args.base_model_names:
    args.inference_model_names.append('TRANSSIG-NLL-NAR')


#import ipdb ; ipdb.set_trace()
if args.dataset_name == 'ett':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 192
    if args.N_output == -1: args.N_output = 192
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_ett_d192_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    if args.output_dir is None:
        args.output_dir = 'Outputs_ett_d192_klnorm_b24_e192_corrshuffle_bs128_seplayers_nodeczeros_nodecconv_t2v_usefeats_t2vglobal_idx_val20'
    #if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size  == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.t2v_type = 'idx'
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = '15min'

elif args.dataset_name == 'etthourly':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 168
    if args.N_output == -1: args.N_output = 168
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_etthourly_noextrafeats_d168_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_etthourly_noextrafeats_d168_klnorm_b24_pefix_e168_val20_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    #if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.00001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 24
    if args.use_feats == -1: args.use_feats = 1
    #args.print_every = 5 # TODO: Only for aggregate models
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'

elif args.dataset_name == 'Solar':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_Solar_d168_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.output_dir is None:
        args.output_dir = 'Outputs_Solar_d168_normzscore_klnorm_b4_e336_corrshuffle_seplayers_nodeczeros_nodecconv_t2v'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    #if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.0005
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'

elif args.dataset_name == 'electricity':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 336
    if args.N_output == -1: args.N_output = 168
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_electricity'
    if args.output_dir is None:
        args.output_dir = 'Outputs_electricity'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    #if args.normalize is None: args.normalize = 'min_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32
    args.freq = 'h'

elif args.dataset_name == 'aggtest':
    if args.epochs == -1: args.epochs = 20
    if args.N_input == -1: args.N_input = 20
    if args.N_output == -1: args.N_output = 10
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_aggtest_test'
    if args.output_dir is None:
        args.output_dir = 'Outputs_aggtest_test'
    if args.normalize is None: args.normalize = 'zscore_per_series'
    if args.learning_rate == -1.: args.learning_rate = 0.005
    if args.batch_size == -1: args.batch_size = 10
    if args.hidden_size == -1: args.hidden_size = 32
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = args.N_output
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:2'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32


elif args.dataset_name == 'foodinflation':
    if args.epochs == -1: args.epochs = 50
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_foodinflation'
    if args.output_dir is None:
        args.output_dir = 'Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zeroshift_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 128
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01
    if args.kernel_size == -1: args.kernel_size = 10
    if args.nkernel == -1: args.nkernel = 32

elif args.dataset_name == 'foodinflationmonthly':
    if args.epochs == -1: args.epochs = 100
    if args.N_input == -1: args.N_input = 90
    if args.N_output == -1: args.N_output = 30
    if args.saved_models_dir is None:
        args.saved_models_dir = 'saved_models_foodinflation'
    if args.output_dir is None:
        args.output_dir = 'Outputs_foodinflation'
    if args.normalize is None: args.normalize = 'zeroshift_per_series'
    if args.learning_rate == -1: args.learning_rate = 0.0001
    if args.batch_size == -1: args.batch_size = 64
    if args.hidden_size == -1: args.hidden_size = 32
    if args.num_grulstm_layers == -1: args.num_grulstm_layers = 1
    if args.v_dim == -1: args.v_dim = 4
    if args.b == -1: args.b = 4
    if args.use_feats == -1: args.use_feats = 1
    if args.device is None: args.device = 'cuda:1'
    if args.cv_inf == -1: args.cv_inf = 1
    if args.lr_inf == -1: args.lr_inf = 0.01    
    if args.kernel_size == -1: args.kernel_size = 2
    if args.nkernel == -1: args.nkernel = 32


print('Command Line Arguments:')
print(args)

#import ipdb ; ipdb.set_trace()

inference_models = {}
for name in args.inference_model_names:
    inference_models[name] = {}


DUMP_PATH = '/mnt/infonas/data/pratham/Forecasting/TransNAR'
args.output_dir = os.path.join(DUMP_PATH, args.output_dir)
args.saved_models_dir = os.path.join(DUMP_PATH, args.saved_models_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.saved_models_dir, exist_ok=True)


# ----- Start: Load the datasets ----- #

data_processor = utils.DataProcessor(args)
dataset = data_processor.get_processed_data(args)

# ----- End: Load the datasets ----- #


# ----- Start: Models training ----- #
base_models = {}
for base_model_name in args.base_model_names:

    trainloader = dataset['trainloader']
    devloader = dataset['devloader']
    testloader = dataset['testloader']
    feats_info = dataset['feats_info']
    N_input = dataset['N_input']
    N_output = dataset['N_output']
    input_size = dataset['input_size']
    output_size = dataset['output_size']
    dev_norm = dataset['dev_norm']
    test_norm = dataset['test_norm']

    if base_model_name in [
        'rnn-mse-nar', 'rnn-mse-ar', 'trans-mse-nar', 'trans-mse-ar',
        'gpt-mse-ar', 'gpt-mse-nar', 'informer-mse-nar',
        'oracle', 'oracleforecast',
    ]:
        estimate_type = 'point'
    elif base_model_name in [
        'trans-q-nar', 'rnn-q-nar', 'rnn-q-ar',
        'rnn-nll-nar', 'rnn-nll-ar', 'trans-nll-ar',
        'gpt-nll-ar', 'gpt-nll-nar', 'transsig-nll-nar'
    ]:
        estimate_type = 'variance'
    elif base_model_name in [
        'rnn-fnll-nar', 'trans-fnll-ar'
    ]:
        estimate_type = 'covariance'
    elif base_model_name in ['trans-bvnll-ar']:
        estimate_type = 'bivariate'

    saved_models_dir = os.path.join(
        args.saved_models_dir, args.dataset_name+'_'+base_model_name
    )
    os.makedirs(saved_models_dir, exist_ok=True)
    writer = SummaryWriter(saved_models_dir)
    saved_models_path = os.path.join(saved_models_dir, 'state_dict_model.pt')
    print('\n {}'.format(base_model_name))

    # Create the network
    net_gru = get_base_model(
        args, base_model_name, N_input, N_output, input_size, output_size,
        estimate_type, feats_info
    )

    # train the network
    if base_model_name not in ['oracle', 'oracleforecast']:
        train_model(
            args, base_model_name, net_gru,
            dataset, saved_models_path, writer, verbose=1
        )

    base_models[base_model_name] = net_gru

    writer.flush()

writer.close()
# ----- End: base models training ----- #

# ----- Start: Inference models for bottom level----- #
print('\n Starting Inference Models')

#import ipdb
#ipdb.set_trace()


def run_inference_model(args, inf_model_name, base_models, which_split):

    metric2val = dict()
    infmodel2preds = dict()

    if inf_model_name in ['RNN-MSE-NAR']:
        base_models_dict = base_models['rnn-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-NLL-NAR']:
        base_models_dict = base_models['rnn-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-NLL-AR']:
        base_models_dict = base_models['rnn-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-MSE-AR']:
        base_models_dict = base_models['trans-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-NLL-AR']:
        base_models_dict = base_models['trans-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['GPT-NLL-AR']:
        base_models_dict = base_models['gpt-nll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['GPT-MSE-AR']:
        base_models_dict = base_models['gpt-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['GPT-NLL-NAR']:
        base_models_dict = base_models['gpt-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['GPT-MSE-NAR']:
        base_models_dict = base_models['gpt-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['INFORMER-MSE-NAR']:
        base_models_dict = base_models['informer-mse-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-BVNLL-AR']:
        base_models_dict = base_models['trans-bvnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['TRANS-FNLL-AR']:
        base_models_dict = base_models['trans-fnll-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-FNLL-NAR']:
        base_models_dict = base_models['rnn-fnll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['oracle']:
        base_models_dict = base_models['oracle']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['SimRetrieval']:
        base_models_dict = base_models['oracleforecast']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device, is_oracle=True)

    elif inf_model_name in ['TRANSSIG-NLL-NAR']:
        base_models_dict = base_models['transsig-nll-nar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-Q-NAR']:
        base_models_dict = base_models['rnn-q-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['RNN-MSE-AR']:
        base_models_dict = base_models['rnn-mse-ar']
        inf_net = inf_models.RNNNLLNAR(base_models_dict, device=args.device)

    elif inf_model_name in ['RNN-Q-AR']:
        base_models_dict = base_models['rnn-q-ar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-MSE-NAR']:
        base_models_dict = base_models['trans-mse-nar']['sum']
        raise NotImplementedError

    elif inf_model_name in ['TRANS-Q-NAR']:
        base_models_dict = base_models['trans-q-nar']['sum']
        raise NotImplementedError

    if not args.leak_agg_targets:
        inf_test_targets_dict = None

    inf_net.eval()
    (
        inputs, target, pred_mu, pred_std, pred_d, pred_v,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape,
        total_time
    ) = eval_inf_model(args, inf_net, dataset, which_split, args.gamma, verbose=1)

    if inf_net.covariance == False:
        pred_v_foragg = None
    else:
        pred_v_foragg = pred_v
    #import ipdb ; ipdb.set_trace()
    agg2metrics = eval_aggregates(
        inputs, target, pred_mu, pred_std, pred_d, pred_v_foragg
    )

    inference_models[inf_model_name] = inf_net
    metric_mse = metric_mse.item()

    print('Metrics for Inference model {}: MAE:{:f}, CRPS:{:f}, MSE:{:f}, SMAPE:{:f}, Time:{:f}'.format(
        inf_model_name, metric_mae, metric_crps, metric_mse, metric_smape, total_time)
    )

    metric2val = utils.add_metrics_to_dict(
        metric2val, inf_model_name,
        metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae, metric_smape
    )
    infmodel2preds[inf_model_name] = pred_mu
    if which_split in ['test']:
        output_dir = os.path.join(args.output_dir, args.dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        utils.write_arr_to_file(
            output_dir, inf_model_name,
            inputs.detach().numpy(),
            target.detach().numpy(),
            pred_mu.detach().numpy(),
            pred_std.detach().numpy(),
            pred_d.detach().numpy(),
            pred_v.detach().numpy()
        )

    return metric2val, agg2metrics

model2metrics = dict()
for inf_model_name in args.inference_model_names:

    metric2val, agg2metrics = run_inference_model(
        args, inf_model_name, base_models, 'test'
    )
    model2metrics[inf_model_name] = metric2val


with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.txt'), 'w') as fp:

    fp.write('\nModel Name, MAE, DTW, TDI')
    for model_name, metrics_dict in model2metrics.items():
        fp.write(
            '\n{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                model_name,
                metrics_dict['mae'],
                metrics_dict['crps'],
                metrics_dict['mse'],
                metrics_dict['dtw'],
                metrics_dict['tdi'],
            )
        )

for model_name, metrics_dict in model2metrics.items():
    for metric, metric_val in metrics_dict.items():
        model2metrics[model_name][metric] = str(metric_val)
with open(os.path.join(args.output_dir, 'results_'+args.dataset_name+'.json'), 'w') as fp:
    json.dump(model2metrics, fp)

# ----- End: Inference models for bottom level----- #


# ----- Start: Base models for all aggreagations and levels --- #

if False:
    model2metrics = {}
    for base_model_name in args.base_model_names:
        for agg_method in args.aggregate_methods:
            for K in args.K_list:
    
                print('Base Model', base_model_name,'for', agg_method, K)
        
                loader = dataset['testloader']
                norm = dataset['test_norm']
                (
                    test_inputs, test_target, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae, metric_crps_part, metric_nll
                ) = eval_base_model(
                    args, base_model_name, base_models[base_model_name],
                    loader, norm, args.gamma, verbose=1, unnorm=True
                )
    
                output_dir = os.path.join(args.output_dir, args.dataset_name + '_base')
                os.makedirs(output_dir, exist_ok=True)
                utils.write_aggregate_preds_to_file(
                    output_dir, base_model_name, agg_method, K,
                    test_inputs, test_target, pred_mu, pred_std
                )
    
                model2metrics = utils.add_base_metrics_to_dict(
                    model2metrics, agg_method, K, base_model_name,
                    metric_mse, metric_dtw, metric_tdi, metric_crps, metric_mae
                )
    
    
    with open(os.path.join(args.output_dir, 'results_base_'+args.dataset_name+'.txt'), 'w') as fp:
    
        fp.write('\nModel Name, MAE, DTW, TDI')
        for agg_method in model2metrics.keys():
            for K in model2metrics[agg_method].keys():
                for model_name, metrics_dict in model2metrics[agg_method][K].items():
                    fp.write(
                        '\n{}, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                            agg_method, K, model_name,
                            metrics_dict['mae'],
                            metrics_dict['crps'],
                            metrics_dict['mse'],
                            metrics_dict['dtw'],
                            metrics_dict['tdi'],
                        )
                    )
    
    for model_name, metrics_dict in model2metrics.items():
        for metric, metric_val in metrics_dict.items():
            model2metrics[model_name][metric] = str(metric_val)
    with open(os.path.join(args.output_dir, 'results_base_'+args.dataset_name+'.json'), 'w') as fp:
        json.dump(model2metrics, fp)

# ----- End: Base models for all aggreagations and levels --- #
