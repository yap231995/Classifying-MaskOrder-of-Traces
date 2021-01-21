import os
import torch
import numpy as np
from torch.utils.data import Dataset
from src.gen_mask_traces import TraceGenerator
from sklearn import preprocessing, model_selection



class MaskingDataset_train(Dataset):
    """Face Landmarks dataset."""

    def __init__(self ,config, transform=None):
        self.config = config
        self.Generator = TraceGenerator(config)
        self.transform = transform
        # self.n_traces  = n_traces
        # self.traces_mask_order0, self.Y0 = self.Generator.gen_many_trace_mask_order0()
        # self.traces_mask_order1, self.Y1 = self.Generator.gen_many_trace_mask_order1()
        # self.traces_mask_order1_nomaskleak, self.Y1_nomaskleak = self.Generator.gen_many_trace_mask_order1_nomaskleak()
        # self.traces_mask_order2, self.Y2 = self.Generator.gen_many_trace_mask_order2()
        # self.traces_mask_order3, self.Y3 = self.Generator.gen_many_trace_mask_order3()
        self.Mix_traces, _, self.MaskOrders = self.Generator.gen_mix_traces_order_0to3(config.gen_mask_traces.n_traces)

    def __len__(self):
        return len(self.Mix_traces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.sensitive[idx]
        trace = self.Mix_traces[idx]
        maskorder = self.MaskOrders[idx]

        sample = {'trace': trace, 'maskorder': maskorder}

        if self.transform:
            sample = self.transform(sample)

        return sample

    ### preprocessing with respect to the features

    def train_validation_split(self, test_size = 0.1):
        self.Mix_traces, Mix_traces_validation, self.MaskOrders, MaskOrders_validation = model_selection.train_test_split(self.Mix_traces, self.MaskOrders, test_size = test_size, random_state =0)
        return [Mix_traces_validation,MaskOrders_validation]

    def feature_min_max_scaling(self, a, b):
        scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
        self.Mix_traces = scaler.fit_transform(self.Mix_traces)
        self.feature_scaler = scaler

    def feature_standardization(self):
        scaler = preprocessing.StandardScaler()
        self.Mix_traces = scaler.fit_transform(self.Mix_traces)
        self.feature_scaler = scaler

    def get_feature_scaler(self):
        return self.feature_scaler

    # def to_categorical(self, num_classes):
    #     self.Y_profiling = np.eye(num_classes, dtype='uint8')[self.Y_profiling]




class MaskingDataset_validation(Dataset):

    def __init__(self, config, Mix_traces_validation, MaskOrders_validation, transform=None, feature_scaler = None):
        self.config = config
        self.Mix_traces, self.MaskOrders = Mix_traces_validation, MaskOrders_validation
        self.transform = transform
        self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.Mix_traces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.sensitive[idx]
        trace = self.Mix_traces[idx]
        maskorder = self.MaskOrders[idx]

        sample = {'trace': trace, 'maskorder': maskorder}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def feature_scaling(self, feature_scaler = None):
        if self.feature_scaler == None and feature_scaler == None:
            return "No feature scaler"
        elif feature_scaler != None:
            self.Mix_traces = feature_scaler.transform(self.Mix_traces)
        else:
            self.Mix_traces = self.feature_scaler.transform(self.Mix_traces)


class MaskingDataset_test(Dataset):

    def __init__(self, config, transform=None, feature_scaler = None):
        self.config = config
        self.Generator = TraceGenerator(config)
        self.Mix_traces, _, self.MaskOrders = self.Generator.gen_mix_traces_order_0to3(config.test_dataloader.n_traces)
        self.transform = transform
        self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.Mix_traces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # label = self.sensitive[idx]
        trace = self.Mix_traces[idx]
        maskorder = self.MaskOrders[idx]

        sample = {'trace': trace, 'maskorder': maskorder}

        if self.transform:
            sample = self.transform(sample)
        return sample


    # def to_categorical(self, num_classes):
    #     self.targets = np.eye(num_classes, dtype='uint8')[self.targets]


    def feature_scaling(self, feature_scaler = None):
        if self.feature_scaler == None and feature_scaler == None:
            return "No feature scaler"
        elif feature_scaler != None:
            self.Mix_traces = feature_scaler.transform(self.Mix_traces)
        else:
            self.Mix_traces = self.feature_scaler.transform(self.Mix_traces)