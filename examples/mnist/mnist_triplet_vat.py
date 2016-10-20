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
from masalachai import DataFeeder
from masalachai import datafeeders
from masalachai import Logger
from masalachai import trainers
from masalachai import models
from masalachai.optimizer_schedulers import DecayOptimizerScheduler
from mlp import Mlp

# argparse
parser = argparse.ArgumentParser(description='Supervised Multi Layer Perceptron Example')
parser.add_argument('--nitr', '-n', type=int, default=10000, help='number of times of weight update (default: 10000)')
parser.add_argument('--lbatch', '-l', type=int, default=100, help='training batchsize (default: 300)')
parser.add_argument('--ubatch', '-u', type=int, default=250, help='training batchsize (default: 300)')
parser.add_argument('--valbatch', '-v', type=int, default=100, help='validation batchsize (default: 100)')
parser.add_argument('--slabeled', '-s', type=int, default=100, help='validation batchsize (default: 100)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device #, if you want to use cpu, use -1 (default: -1)')
args = parser.parse_args()

def mnist_preprocess(data):
    data['data0'] /= 255.
    data['data1'] /= 255.
    data['data2'] /= 255.
    return data


def mnist_preprocess_u(data):
    data['data'] /= 255.
    return data

# Logger setup
def vat_train_log(res):
    log_str = '{0:d}, loss={1:.5f}, lds={2:.5f}'.format(res['iteration'], res['loss'], res['lds'])
    return log_str

logger = Logger('MNIST TRIPLET MLP VAT',
        train_log_mode='TRAIN_VAT',
        test_log_mode='TEST_LOSS_ONLY')
logger.mode['TRAIN_VAT'] = vat_train_log

# Configure GPU Device
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# loading dataset
dataset = mnist.load()
dim = dataset['train']['data'][0].size
N_train = len(dataset['train']['target'])
N_test = len(dataset['test']['target'])
test_data_dict = {'data':dataset['test']['data'][:1000].reshape(1000, dim).astype(np.float32),
        'target':dataset['test']['target'][:1000].astype(np.int32)}
unlabeled_data_dict = {'data':dataset['train']['data'].reshape(N_train, dim).astype(np.float32)}

# making labeled data
lplo = cross_validation.LeavePLabelOut(labels=six.moves.range(N_train), p=args.slabeled)
fold = 1
for i in six.moves.range(fold):
    train_idx, test_idx = next(iter(lplo))
labeled_data_dict = {'data':unlabeled_data_dict['data'][test_idx].astype(np.float32),
                     'target':dataset['train']['target'][test_idx].astype(np.int32)}

labeled_data = datafeeders.TripletFeeder(labeled_data_dict, batchsize=args.lbatch)
unlabeled_data = DataFeeder(unlabeled_data_dict, batchsize=args.ubatch)
test_data = datafeeders.TripletFeeder(test_data_dict, batchsize=args.valbatch)

labeled_data.hook_preprocess(mnist_preprocess)
unlabeled_data.hook_preprocess(mnist_preprocess_u)
test_data.hook_preprocess(mnist_preprocess)


# Model Setup
h_units = 1200
model = models.TripletModel(Mlp(labeled_data['data'][0].size, h_units, h_units, np.max(labeled_data['target'])+1))


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Opimizer Setup
alpha=0.002
optimizer = optimizers.Adam(alpha=alpha)
optimizer.setup(model)

alpha_decay_interval = 500
alpha_decay_rate = 0.9
adam_alpha_scheduler = DecayOptimizerScheduler(optimizer, 'alpha', alpha_decay_interval, alpha_decay_rate)

trainer = trainers.VirtualAdversarialTrainer(optimizer, 
                                             logger, 
                                             (labeled_data, unlabeled_data), 
                                             test_data, 
                                             args.gpu, 
                                             norm='euclidean_d',
                                             eps=1.4, xi=10, lam=1.0)
trainer.train(args.nitr, 
              log_interval=1,
              test_interval=100, 
              test_nitr=10)
