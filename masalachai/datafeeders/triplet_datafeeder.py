# -*- coding: utf-8 -*-

import six
import multiprocessing
import numpy
import itertools

import masalachai
from masalachai.datafeeder import DataFeeder

def triplet_preprocess(inp):
    data = {}
    data0, data1, data2 = inp
    data['data0'] = data0['data']
    data['data1'] = data1['data']
    data['data2'] = data2['data']
    return data

class TripletFeeder(DataFeeder):
    def __init__(self, data_dict, batchsize=1, shuffle=True, loaderjob=8):
        super(TripletFeeder, self).__init__(data_dict, batchsize, shuffle, loaderjob)
        self.hook_preprocess(triplet_preprocess)
        self.xa = None
        self.xp = None
        self.xn = None

    def run(self):
        pool = multiprocessing.Pool(self.loaderjob)
        while not self.stop.is_set():
            n_class = len(list(set(self.data_dict['target'])))
            nl_class = [len(numpy.where(self.data_dict['target'] == c)[0]) 
                    for c in range(n_class)]
            perm = numpy.random.permutation(self.n) if self.shuffle \
                    else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n and not self.stop.is_set():
                indexes = [next(gen) for b in six.moves.range(0, self.batchsize)]
                classwise_gens = [itertools.cycle(numpy.random.permutation(numpy.where(self.data_dict['target']==c)[0])[:self.batchsize])
                        for c in six.moves.range(n_class)]
                classwise_batch = [[next(g) for b in six.moves.range(self.batchsize)] for g in classwise_gens]

                xa = self.get_data_dict_list(indexes)
                xa_targets = [self.data_dict['target'][idx] for idx in indexes]

                xp = self.get_data_dict_list(numpy.array([classwise_batch[t].pop() for t in xa_targets]))
                xn = self.get_data_dict_list(
                        numpy.asarray([classwise_batch[numpy.random.choice(
                            n_class,1,p=[1./(n_class-1) if c!=t else 0.0 for c in six.moves.range(n_class)])[0]].pop()
                        for t in xa_targets]))

                batch_pool = []
                for i in range(len(indexes)):
                    batch_pool.append(
                            pool.apply_async(
                                masalachai.datafeeder.preprocess, (
                                    self.preprocess_hooks, (
                                        xa[i],
                                        xp[i],
                                        xn[i]))))
                self.queue.put(self.get_data_dict_from_list(
                    [p.get() for p in batch_pool]))
                cnt += self.batchsize
        pool.close()
        pool.join()
