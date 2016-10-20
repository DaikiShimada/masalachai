#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers
from sklearn import cross_validation

# import dataset script
import mnist

# import model network
from masalachai import models
from masalachai import datafeeders
from masalachai import Logger
from masalachai import trainers
from convnet import ConvNet

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--nitr', '-n', type=int, default=1000, help='number of times of weight update (default: 1000)')
parser.add_argument('--batch', '-b', type=int, default=10, help='training batchsize (default: 100)')
parser.add_argument('--valbatch', '-v', type=int, default=10, help='validation batchsize (default: 1000)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preprocess(data):
    data['data0'] /= 255.
    data['data1'] /= 255.
    data['data2'] /= 255.

    data['data0'] = np.expand_dims(data['data0'], 0)
    data['data1'] = np.expand_dims(data['data1'], 0)
    data['data2'] = np.expand_dims(data['data2'], 0)
    return data

# Logger setup
logger = Logger('MNIST TRIPLET',
                train_log_mode='TRAIN_LOSS_ONLY',
                test_log_mode='TEST_LOSS_ONLY')

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
N_train = len(dataset['train']['target'])
N_test = len(dataset['test']['target'])
train_data_dict = {'data':dataset['train']['data'].astype(np.float32),
                 'target':dataset['train']['target'].astype(np.int32)}
test_data_dict = {'data':dataset['test']['data'].astype(np.float32),
                  'target':dataset['test']['target'].astype(np.int32)}
logger('load data')

train_data = datafeeders.TripletFeeder(train_data_dict, batchsize=args.batch)
test_data =  datafeeders.TripletFeeder(test_data_dict, batchsize=args.valbatch)

train_data.hook_preprocess(mnist_preprocess)
test_data.hook_preprocess(mnist_preprocess)


# Model Setup
outputs = 2
model = models.TripletModel(ConvNet(output=outputs))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Opimizer Setup
optimizer = optimizers.SGD(0.01)
optimizer.setup(model)


trainer = trainers.SupervisedTrainer(optimizer, logger, (train_data,), None, args.gpu)
trainer.train(args.nitr, 1, 100, 1)
