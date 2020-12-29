import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.dilate_loss import dilate_loss
from eval import eval_base_model
import time
from models.base_models import get_base_model

import optuna
from ray import tune


def get_optimizer(args, config, net):
    lr = config['lr']
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)
    return optimizer, scheduler


def train_model(config):

    lr = config["lr"]
    args = config['args']
    model_name = config['base_model_name']
    trainloader = config['trainloader']
    devloader = config['devloader']
    testloader = config['testloader']
    norm = config['dev_norm']
    saved_models_path = config['saved_models_path']
    output_dir = config['output_dir']
    #writer = config['writer']
    eval_every = config['eval_every']
    verbose = config['verbose']
    level = config['level']
    N_input = config['N_input']
    N_output = config['N_output']
    input_size = config['input_size']
    output_size = config['output_size']
    point_estimates = config['point_estimates']
    Lambda=1


    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Create the network
    net = get_base_model(
        args, config, level,
        N_input, N_output, input_size, output_size,
        point_estimates
    )
    optimizer, scheduler = get_optimizer(args, config, net)

    criterion = torch.nn.MSELoss()

    #optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)
    #optimizer = torch.optim.SGD(net.parameters(),lr=args.learning_rate)
    if False:
    #if (not args.ignore_ckpt) and os.path.isfile(saved_models_path):
        print('Loading from saved model')
        checkpoint = torch.load(saved_models_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_metric = checkpoint['metric']
        epochs = 0
    else:
        if args.ignore_ckpt:
            print('Ignoring saved checkpoint')
        else:
            print('No saved model found')
        best_epoch = -1
        best_metric = np.inf
        epochs = args.epochs
    net.train()

    for curr_epoch in range(best_epoch+1, best_epoch+1+epochs):
        epoch_loss = 0.
        for i, data in enumerate(trainloader, 0):
            #st = time.time()
            inputs, target, feats_in, feats_tgt, _, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
            target = torch.tensor(target, dtype=torch.float32).to(args.device)
            feats_in = torch.tensor(feats_in, dtype=torch.float32).to(args.device)
            feats_tgt = torch.tensor(feats_tgt, dtype=torch.float32).to(args.device)
            batch_size, N_output = target.shape[0:2]

            # forward + backward + optimize
            means, stds = net(feats_in, inputs, feats_tgt, target)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)

            if model_name in ['seq2seqmse']:
                loss_mse = criterion(target,means)
                loss = loss_mse
            if model_name in ['seq2seqdilate']:
                loss, loss_shape, loss_temporal = dilate_loss(target, means, args.alpha, args.gamma, args.device)
            if model_name in ['seq2seqnll']:
                if args.train_twostage:
                    if curr_epoch < epochs/2:
                        stds = torch.ones_like(stds)
                    if curr_epoch-1 <= epochs/2 and curr_epoch > epochs/2:
                        best_metric = np.inf
                dist = torch.distributions.normal.Normal(means, stds)
                loss = -torch.mean(dist.log_prob(target))

                if args.mse_loss_with_nll:
                    loss += criterion(target, means)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #et = time.time()
            #print('Time required for batch ', i, ':', et-st, 'loss:', loss.item())
            if i>=100:
                break

        # ...log the epoch_loss
        #if model_name in ['seq2seqdilate']:
        #    writer.add_scalar('training_loss/DILATE', epoch_loss, curr_epoch)
        #if model_name in ['seq2seqmse']:
        #    writer.add_scalar('training_loss/MSE', epoch_loss, curr_epoch)
        #if model_name in ['seq2seqnll']:
        #    writer.add_scalar('training_loss/NLL', epoch_loss, curr_epoch)


        if(verbose):
            if (curr_epoch % args.print_every == 0):
                print('curr_epoch ', curr_epoch, \
                      ' epoch_loss ', epoch_loss, \
                      ' loss shape ',loss_shape.item(), \
                      ' loss temporal ',loss_temporal.item(), \
                      'learning_rate:', optimizer.param_groups[0]['lr'])
                (
                    _, _, pred_mu, pred_std,
                    metric_dilate, metric_mse, metric_dtw, metric_tdi,
                    metric_crps, metric_mae, metric_crps_part
                )= eval_base_model(
                    args, model_name, net, devloader, norm, args.gamma, verbose=1
                )

                if model_name in ['seq2seqdilate']:
                    metric = metric_dilate
                else:
                    metric = metric_crps

                if metric < best_metric:
                    best_metric = metric
                    best_epoch = curr_epoch
                #    state_dict = {
                #                'model_state_dict': net.state_dict(),
                #                'optimizer_state_dict': optimizer.state_dict(),
                #                'epoch': best_epoch,
                #                'metric': best_metric,
                #                }
                #    torch.save(state_dict, saved_models_path)
                #    print('Model saved at epoch', curr_epoch)

                scheduler.step(metric)

                # ...log the metrics
                #if model_name in ['seq2seqdilate']:
                #    writer.add_scalar('dev_metrics/dilate', metric_dilate, curr_epoch)
                #writer.add_scalar('dev_metrics/crps', metric_crps, curr_epoch)
                #writer.add_scalar('dev_metrics/mae', metric_mae, curr_epoch)
                #writer.add_scalar('dev_metrics/mse', metric_mse, curr_epoch)

                #trial.report(metric, curr_epoch)
                #if trial.should_prune():
                #    raise optuna.exceptions.TrialPruned()

                with tune.checkpoint_dir(curr_epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    print('CHECKPOINT DIR:', path)
                    state_dict = {
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': best_epoch,
                                'metric': best_metric,
                                }
                    torch.save(state_dict, path)

                tune.report(metric=metric)


#    print('Best model found at epoch', best_epoch)
#    #net.load_state_dict(torch.load(saved_models_path))
#    #checkpoint = torch.load(saved_models_path)
#    #net.load_state_dict(checkpoint['model_state_dict'])
#    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    net.eval()
#    (
#        _, _, pred_mu, pred_std,
#        metric_dilate, metric_mse, metric_dtw, metric_tdi,
#        metric_crps, metric_mae, metric_crps_part
#    ) = eval_base_model(
#        args, model_name, net, devloader, norm, args.gamma,verbose=1
#    )
#
#    if model_name in ['seq2seqdilate']:
#        metric = metric_dilate
#    else:
#        metric = metric_crps
#
#    return metric
