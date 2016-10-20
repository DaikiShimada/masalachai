# -*- coding: utf-8 -*-

import six
import threading
import multiprocessing
import numpy
import pandas
import itertools

def from_csv(filename, batchsize=1, shuffle=True, loaderjob=8):
    df = pandas.read_csv(fname)
    return DataFeeder(df.to_dict(), batchsize=batchsize, shuffle=shuffle, loaderjob=loaderjob)


def preprocess(preprocess_hooks, data):
    for p in preprocess_hooks:
        data = p(data)
    return data


class DataFeeder(object):
    # data_dict has following keys: 'data' and 'target'.

    def __init__(self, data_dict, batchsize=1, shuffle=True, loaderjob=8):
        super(DataFeeder, self).__init__()

        self.data_dict = data_dict
        self.n = len(data_dict['data'])

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.loaderjob = loaderjob

        self.preprocess_hooks = []
        self.thread = threading.Thread(target=self.run)
        self.thread.setDaemon(True)
        self.queue = None
        self.stop = None

    def __getitem__(self, index):
        return self.data_dict[index]

    def hook_preprocess(self, func):
        self.preprocess_hooks.append(func)

    def setQueue(self, queue):
        self.queue = queue

    def setEvent(self, event):
        self.stop = event

    def generateThread(self, queue, event):
        self.queue = queue
        self.stop = event
        self.thread = threading.Thread(target=self.run)
        self.thread.setDaemon(True)

    def get_data_dict_list(self, indexes):
        return [{k: v[i] if getattr(v, '__iter__', False) and len(v) == self.n
                 else v for k, v in self.data_dict.items()} for i in indexes]

    def get_data_dict_from_list(self, dict_list):
        return {k: numpy.stack([d[k] for d in dict_list])
                for k in dict_list[0].keys() if 'data' in k or 'target' in k}

    def run(self):
        pool = multiprocessing.Pool(self.loaderjob)
        while not self.stop.is_set():
            perm = numpy.random.permutation(
                self.n) if self.shuffle else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n and not self.stop.is_set():
                indexes = [next(gen)
                           for b in six.moves.range(0, self.batchsize)]
                data_dict_list = self.get_data_dict_list(indexes)
                batch_pool = []
                for i in range(len(indexes)):
                    batch_pool.append(pool.apply_async(
                        preprocess, (self.preprocess_hooks,
                                     data_dict_list[i])))

                self.queue.put(self.get_data_dict_from_list(
                    [p.get() for p in batch_pool]))
                cnt += self.batchsize
        try:
            pool.close()
        except:
            pool.terminate()
            raise

    def batch(self, batchsize):
        while True:
            perm = numpy.random.permutation(
                self.n) if self.shuffle else numpy.arange(self.n)
            gen = itertools.cycle(perm)
            cnt = 0
            while cnt < self.n:
                indexes = [next(gen) for b in six.moves.range(0, batchsize)]
                cnt += batchsize
                yield {'data': (self.data_dict['data'][indexes],),
                       'target': (self.data_dict['target'][indexes],)}
