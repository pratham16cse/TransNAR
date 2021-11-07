import os
import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose, STL

if os.path.exists('bee'):
    DATA_DIRS = '/mnt/infonas/data/pratham/Forecasting/DILATE'
else:
    DATA_DIRS = '.'

def get_list_of_dict_format(data):
    data_new = list()
    for entry in data:
        entry_dict = dict()
        entry_dict['target'] = entry
        data_new.append(entry_dict)
    return data_new

def prune_dev_test_sequence(data, seq_len):
    for i in range(len(data)):
        data[i]['target'] = data[i]['target'][-seq_len:]
        data[i]['feats'] = data[i]['feats'][-seq_len:]
    return data


def parse_ett(dataset_name, N_input, N_output, t2v_type=None):
    df = pd.read_csv(os.path.join(DATA_DIRS, 'data', 'ETT', 'ETTm1.csv'))
    # Remove incomplete data from last day
    df = df[:-80]


    data = df[['OT']].to_numpy().T
    #data = np.expand_dims(data, axis=-1)

    n = data.shape[1]
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len
    #train_len = int(0.7*n)
    #dev_len = (int(0.2*n)//N_output) * N_output
    #test_len = n - train_len - dev_len

    #import ipdb ; ipdb.set_trace()

    feats_cont = np.expand_dims(df[['HUFL','HULL','MUFL','MULL','LUFL','LULL']].to_numpy(), axis=0)
    #feats = ((feats - np.mean(feats, axis=0, keepdims=True)) / np.std(feats, axis=0, keepdims=True))
    #feats = np.ones((data.shape[0], 1)) * np.expand_dims(np.arange(n), axis=0) % 60
    #feats_discrete = np.abs((np.ones((data.shape[0], 1)) * np.expand_dims(np.arange(n), axis=0) % 60) // 15)
    feats_discrete = np.abs((np.ones((data.shape[0], 1)) * np.expand_dims(np.arange(n), axis=0) % 24*4))
    feats_discrete = np.expand_dims(feats_discrete, axis=-1)

    cal_date = pd.to_datetime(df['date'])
    #import ipdb ; ipdb.set_trace()
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                #cal_date.dt.year,
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
                #cal_date.dt.minute
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    #import ipdb ; ipdb.set_trace()
    feats_date = np.expand_dims(feats_date, axis=0)

    #import ipdb ; ipdb.set_trace()
    feats_month = np.expand_dims(np.expand_dims(cal_date.dt.month-1, axis=-1), axis=0)

    #feats = np.concatenate([feats_discrete, feats_cont], axis=-1)
    #feats = feats_cont
    #feats = np.concatenate([feats_cont, feats_date], axis=-1)
    feats = np.concatenate([feats_discrete, feats_cont, feats_date], axis=-1)
    #feats = np.concatenate([feats_discrete, feats_month, feats_cont, feats_date], axis=-1)

    #data = (data - np.mean(data, axis=0, keepdims=True)).T

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    #import ipdb ; ipdb.set_trace()

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = {}, {}
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map[len(data_dev)-1] = i
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map[len(data_test)-1] = i


    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(24*4, 64), 1:(0, 1), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1), 6:(0, 1)}
    #feats_info = {
    #    0:(0, 1), 1:(0, 1), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1),
    #}
    #feats_info = {0:(24*4, 32), 1:(12, 8), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1), 6:(0, 1), 7:(0, 1)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )


def parse_Solar(dataset_name, N_input, N_output, t2v_type=None):

    data, feats = [], []
    with open(os.path.join(DATA_DIRS, 'data', 'solar_nips', 'train', 'train.json')) as f:
        for line in f:
            line_dict = json.loads(line)
            x = line_dict['target']
            data.append(x)
            n = len(x)
            x_f = np.expand_dims((np.arange(len(x)) % 24), axis=-1)
            cal_date = pd.date_range(
                start=line_dict['start'], periods=len(line_dict['target']), freq='H'
            ).to_series()
            if t2v_type is None:
                feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
            x_f = np.concatenate([x_f, feats_date], axis=-1)
            feats.append(x_f)

    data_test, feats_test = [], []
    with open(os.path.join(DATA_DIRS, 'data', 'solar_nips', 'test', 'test.json')) as f:
        for line in f:
            line_dict = json.loads(line)
            x = line_dict['target']
            x = np.array(x)
            #x = np.expand_dims(x, axis=-1)
            data_test.append(torch.tensor(x, dtype=torch.float))
            x_f = np.expand_dims((np.arange(len(x)) % 24), axis=-1)
            n = len(x)
            cal_date = pd.date_range(
                start=line_dict['start'], periods=len(line_dict['target']), freq='H'
            ).to_series()
            if t2v_type is None:
                feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
            x_f = np.concatenate([x_f, feats_date], axis=-1)
            feats_test.append(torch.tensor(x_f, dtype=torch.float))

    # Select only last rolling window from test data
    m = len(data)
    data_test, feats_test = data_test[-m:], feats_test[-m:]

    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float)

    # Features
    feats = torch.tensor(feats, dtype=torch.float)#.unsqueeze(dim=-1)

    n = data.shape[1]
    #train_len = int(0.9*n)
    #dev_len = int(0.1*n)
    dev_len = 24*7
    train_len = n - dev_len
    #test_len = data_test.shape[1]

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev = []
    feats_dev = []
    dev_tsid_map= {}
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, n+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map[len(data_dev)-1] = i
    #for i in range(len(data_test)):
    #    for j in range(n+N_output, n+1, N_output):
    #        if j <= len(data_test[i]):
    #            data_test.append(data_test[i, :j])
    #            feats_test.append(feats_test[i, :j])
    #            test_tsid_map[len(data_test)-1] = i % len(data)
    test_tsid_map = {}
    for i in range(len(data_test)):
        test_tsid_map[i] = i % len(data)

    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    #import ipdb;ipdb.set_trace()
    # Only consider last (N_input+N_output)-length chunk from dev and test data
    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)
    #import ipdb;ipdb.set_trace()
                
    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )

def parse_etthourly(dataset_name, N_input, N_output, t2v_type=None):

#    train_len = 52*168
#    dev_len = 17*168
#    test_len = 17*168
#    n = train_len + dev_len + test_len
#    df = pd.read_csv('../Informer2020/data/ETT/ETTh1.csv').iloc[:n]

    df = pd.read_csv(os.path.join(DATA_DIRS, 'data', 'ETT', 'ETTh1.csv'))
    # Remove incomplete data from last day
    df = df[:-20]

    data = df[['OT']].to_numpy().T
    #data = np.expand_dims(data, axis=-1)

    n = data.shape[1]
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    #train_len = int(0.6*n)
    #dev_len = int(0.2*n)
    #test_len = n - train_len - dev_len

    feats_cont = np.expand_dims(df[['HUFL','HULL','MUFL','MULL','LUFL','LULL']].to_numpy(), axis=0)

    cal_date = pd.to_datetime(df['date'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    #feats = np.concatenate([feats_discrete, feats_cont], axis=-1)
    #feats = feats_discrete
    feats = np.concatenate([feats_hod, feats_cont, feats_date], axis=-1)

    #data = (data - np.mean(data, axis=0, keepdims=True)).T

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                print(i,j,n)
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)


    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)


    decompose_type = 'STL'
    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(24, 16), 1:(0, 1), 2:(0, 1), 3:(0, 1), 4:(0, 1), 5:(0, 1), 6:(0, 1)}
    #feats_info = {0:(24, 1)}
    #feats_info = {0:(0, 1)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)
    #import ipdb ; ipdb.set_trace()

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )


def parse_aggtest(dataset_name, N_input, N_output, t2v_type=None):
    n = 500
    data = np.expand_dims(np.arange(n), axis=0)
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    cal_date = pd.date_range(
        start='2015-01-01 00:00:00', periods=n, freq='H'
    ).to_series()
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)
    feats = np.concatenate([feats_hod, feats_date], axis=-1)

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)

    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map,
        feats_info
    )

def parse_electricity(dataset_name, N_input, N_output, t2v_type=None):
    #df = pd.read_csv('data/electricity_load_forecasting_panama/continuous_dataset.csv')
    df = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'continuous_dataset.csv')
    )
    data = df[['nat_demand']].to_numpy().T

    #n = data.shape[1]
    n = (1903 + 1) * 24 # Select first n=1904*24 entries because of non-stationarity in the data after first n values
    data = data[:, :n]
    df = df.iloc[:n]


    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    #import ipdb ; ipdb.set_trace()

    cal_date = pd.to_datetime(df['datetime'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    feats = np.concatenate([feats_hod, feats_date], axis=-1)

    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)

    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )

def parse_foodinflation(dataset_name, N_input, N_output, t2v_type=None):
    #df = pd.read_csv('data/electricity_load_forecasting_panama/continuous_dataset.csv')
    df = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'foodinflation', 'train_data.csv')
    )
    #df['date'] = pd.to_datetime(df['date'])
    data = df[df.columns[1:]].to_numpy().T

    m, n = data.shape[0], data.shape[1]

    #units = n//N_output
    #dev_len = int(0.2*units) * N_output
    #test_len = int(0.2*units) * N_output
    test_len = 30*3
    dev_len = 30*12
    train_len = n - dev_len - test_len

    #import ipdb ; ipdb.set_trace()

    cal_date = pd.to_datetime(df['date'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.tile(np.expand_dims(feats_date, axis=0), (m, 1, 1))

    feats_day = np.expand_dims(np.expand_dims(cal_date.dt.day.values-1, axis=-1), axis=0)
    feats_day = np.tile(feats_day, (m, 1, 1))
    feats_month = np.expand_dims(np.expand_dims(cal_date.dt.month.values-1, axis=-1), axis=0)
    feats_month = np.tile(feats_month, (m, 1, 1))
    feats_dow = np.expand_dims(np.expand_dims(cal_date.dt.dayofweek.values, axis=-1), axis=0)
    feats_dow = np.tile(feats_dow, (m, 1, 1))

    feats_tsid = np.expand_dims(np.expand_dims(np.arange(m), axis=1), axis=2)
    feats_tsid = np.tile(feats_tsid, (1, n, 1))

    #import ipdb ; ipdb.set_trace()

    #feats = np.concatenate([feats_day, feats_month, feats_dow, feats_date], axis=-1)
    #feats = np.concatenate([feats_day, feats_dow, feats_tsid, feats_date], axis=-1)
    feats = np.concatenate([feats_day, feats_dow, feats_date], axis=-1)
    #feats = np.concatenate([feats_day, feats_dow, feats_tsid, feats_date], axis=-1)


    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    #import ipdb ; ipdb.set_trace()

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)

    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    feats_info = {0:(31, 16), 1:(12, 6)}
    feats_info = {0:(31, 16), 1:(7, 6)}
    #feats_info = {0:(31, 16), 1:(7, 6), 2:(m, -2)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)

    #import ipdb ; ipdb.set_trace()

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )

def parse_foodinflationmonthly(dataset_name, N_input, N_output, t2v_type=None):
    #df = pd.read_csv('data/electricity_load_forecasting_panama/continuous_dataset.csv')
    df = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'foodinflation', 'train_data.csv')
    )
    
    df['date'] = pd.to_datetime(df['date'])
    df_monthly = df.set_index(df['date'])
    del df_monthly['date']
    agg_dict = {}
    for food in df.columns[1:]:
        agg_dict[food] = 'mean'
    df_monthly = df_monthly.groupby(pd.Grouper(freq='M')).agg(agg_dict)
    df_monthly.insert(0, 'date', df_monthly.index)
    df_monthly = df_monthly.set_index(np.arange(df_monthly.shape[0]))
    #print(df_monthly)
    df = df_monthly
    
    #df['date'] = pd.to_datetime(df['date'])
    data = df[df.columns[1:]].to_numpy().T

    m, n = data.shape[0], data.shape[1]

    #units = n//N_output
    #dev_len = int(0.2*units) * N_output
    #test_len = int(0.2*units) * N_output
    test_len = 3
    dev_len = 6
    train_len = n - dev_len - test_len

    #import ipdb ; ipdb.set_trace()

    cal_date = pd.to_datetime(df['date'])
    feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.tile(np.expand_dims(feats_date, axis=0), (m, 1, 1))

    feats_day = np.expand_dims(np.expand_dims(cal_date.dt.day.values-1, axis=-1), axis=0)
    feats_day = np.tile(feats_day, (m, 1, 1))
    feats_month = np.expand_dims(np.expand_dims(cal_date.dt.month.values-1, axis=-1), axis=0)
    feats_month = np.tile(feats_month, (m, 1, 1))
    feats_dow = np.expand_dims(np.expand_dims(cal_date.dt.dayofweek.values, axis=-1), axis=0)
    feats_dow = np.tile(feats_dow, (m, 1, 1))

    feats_tsid = np.expand_dims(np.expand_dims(np.arange(m), axis=1), axis=2)
    feats_tsid = np.tile(feats_tsid, (1, n, 1))

    #import ipdb ; ipdb.set_trace()

    #feats = np.concatenate([feats_day, feats_month, feats_dow, feats_date], axis=-1)
    #feats = np.concatenate([feats_day, feats_dow, feats_tsid, feats_date], axis=-1)
    feats = np.concatenate([feats_month, feats_date], axis=-1)
    #feats = np.concatenate([feats_month, feats_tsid, feats_date], axis=-1)


    data = torch.tensor(data, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    feats_train = feats[:, :train_len]

    #import ipdb ; ipdb.set_trace()

    data_dev, data_test = [], []
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)

    data_train = get_list_of_dict_format(data_train)
    data_dev = get_list_of_dict_format(data_dev)
    data_test = get_list_of_dict_format(data_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]

    #feats_info = {0:(31, 16), 1:(12, 6)}
    #feats_info = {0:(31, 16), 1:(7, 6), 2:(m, 16)}
    feats_info = {0:(31, 16)}
    #feats_info = {0:(31, 16), 1:(m, -2)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output
    data_dev = prune_dev_test_sequence(data_dev, seq_len)
    data_test = prune_dev_test_sequence(data_test, seq_len)

    #import ipdb ; ipdb.set_trace()

    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )
