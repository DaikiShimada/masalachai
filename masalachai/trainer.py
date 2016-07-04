import time
import numpy
import six
import chainer
import threading
from chainer import cuda

from masalachai.logger import Logger
from masalachai.datafeeder import DataFeeder


def res_dic_add(dic0, dic1):
    dic1_items = dic1.items()
    for k, v in dic1_items:
        if k in dic0:
            if k is not None and dic0[k] is not None:
                dic0[k] += v
        else:
            dic0[k] = v
    return dic0


def res_dic_mul(dic, value):
    for k in dic.keys():
        if dic[k] is not None:
            dic[k] *= value
    return dic


class Trainer(object):
    queue_size = 3

    def __init__(self, optimizer, logger, train_data_feeders, test_data_feeder=None, gpu=-1):
        self.optimizer = optimizer
        self.gpu = gpu

        # for train data feeder
        self.train_data_feeders = train_data_feeders
        self.train_data_queues = [six.moves.queue.Queue(self.queue_size) for i in six.moves.range(len(self.train_data_feeders))]

        # for test data feeder
        if test_data_feeder is not None:
            self.test_data_feeder = test_data_feeder
            self.test_data_queue = six.moves.queue.Queue(self.queue_size)

        # for logger
        self.logger = logger
        self.log_queue = six.moves.queue.Queue()
        self.logger.setQueue(self.log_queue)

        self._optimizer_param_schedulers = []


    def add_optimizer_scheduler(self, s):
        # Optimizer のパラメータスケジューラの登録
        self._optimizer_param_schedulers.append(s)


    def optimizer_param_process(self, t):
        # Optimizer のパラメータスケジューラの駆動
        for s in self._optimizer_param_schedulers:
            self.optimizer.__dict__[s.param_name] = s.next(t)


    def wait_train_data_queues(self, t=0.05):
        while True:
            empty_flag = False
            for q in self.train_data_queues:
                if q.empty():
                    empty_flag = True
            if not empty_flag:
                break
            time.sleep(t)

    def wait_test_data_queue(self, t=0.05):
        while self.test_data_queue.empty():
            time.sleep(t)


    def predict(self, batchsize, train=False):
        # This method will be called when TEST phase
        raise NotImplementedError


    def update(self, batchsize):
        # This method will be called when TRAIN phase
        raise NotImplementedError


    def train(self, nitr, batchsizes, 
              log_interval=1, 
              test_interval=1, test_nitr=1, test_batchsize=1):

        # setting batchsize of datafeeders
        for tdf, batchsize in zip(self.train_data_feeders, batchsizes):
            tdf.batchsize = batchsize
        if hasattr(self, 'test_data_feeder'):
            self.test_data_feeder.batchsize = test_batchsize
        
        # setting stop event for data feeder and start threads
        stop_feeding = threading.Event()
        for tdf, q in zip(self.train_data_feeders, self.train_data_queues):
            tdf.generateThread(q, stop_feeding)
            tdf.thread.start()

        # start logger threads
        self.logger.start()

        self.wait_train_data_queues()
        train_res = self.update()
        for i in six.moves.range(1, int(nitr)):
            # update
            self.wait_train_data_queues()
            train_res = res_dic_add(train_res, self.update())
            self.optimizer_param_process(i)

            # logging
            if i % log_interval == 0:
                train_res = res_dic_mul(train_res, 1./log_interval)
                train_res['iteration'] = i
                self.log_queue.put(self.logger.train_log_mode)
                self.log_queue.put(train_res)
                train_res = {}
            # test
            if self.test_data_feeder is not None and i % test_interval == 0:
                test_res = self.test(test_nitr)
                self.log_queue.put(self.logger.test_log_mode)
                self.log_queue.put(test_res)

        # end of training
        stop_feeding.set()

        # clearn data queue
        for q in self.train_data_queues:
            while not q.empty():
                q.get()

        self.log_queue.put('END')
        for tdf in self.train_data_feeders:
            tdf.thread.join()
        self.logger.join()


    def test(self, nitr):

        # setting stop event for data feeder
        stop_feeding = threading.Event()
        self.test_data_feeder.generateThread(self.test_data_queue, stop_feeding)

        # start threads
        self.test_data_feeder.thread.start()

        self.wait_test_data_queue()
        test_res = self.predict()
        for i in six.moves.range(1, int(nitr)):
            self.wait_test_data_queue()
            test_res_ = res_dic_add(test_res, self.predict())
        test_res = res_dic_mul(test_res, 1./nitr)

        # end of testing
        stop_feeding.set()

        # clearn data queue
        while not self.test_data_queue.empty():
            self.test_data_queue.get()

        self.test_data_feeder.thread.join()
        return test_res

