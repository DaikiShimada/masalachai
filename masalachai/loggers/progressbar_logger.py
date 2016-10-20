# -*- coding: utf-8 -*-

from masalachai.logger import Logger
from masalachai.loggers.masalachai_progressbar import MasalachaiProgressBar
import logging
import threading
import progressbar

def diff_from_history(now_dict, history_dict, history_len):
    # history_dict = {'loss': [0.1, 0.2, 0.1], ...}
    ave_dict = {k: sum(v)/float(len(v)) for k, v in history_dict.items()}
    diff_dict = {}

    for k in now_dict.keys():
        #key = now_dict['type']+'_'+k
        key = k
        diff_dict['diff_'+key] = now_dict[key] - ave_dict[key] if key in ave_dict else 0.0
        if history_dict.has_key(k) and len(history_dict[k]) > history_len:
            history_dict[k].pop()
        if k in history_dict:
            history_dict[k].append(now_dict[k])
        else:
            history_dict[k] = [now_dict[k]]

    return diff_dict


class ProgressbarLogger(Logger):
     
    def __init__(self, name, max_value=100, history_len=5, display=True,
            display_data={'train':['loss', 'accuracy'], 'test':['loss', 'accuracy']},
            level=logging.INFO, train_log_mode='TRAIN_PROGRESS', test_log_mode='TEST_PROGRESS'):
        super(ProgressbarLogger, self).__init__(
                name, level=level, display=display, logfile=None,
                train_log_mode=train_log_mode, test_log_mode=test_log_mode)

        self.train_log_data = {}
        self.test_log_data = {}
        self.max_value = max_value
        self.history_len = history_len
        self.display_data = display_data
        self.mode['TRAIN_PROGRESS'] = self.log_train_progress
        self.mode['TEST_PROGRESS'] = self.log_test_progress

        # create logging format
        self.widgets = [progressbar.FormatLabel('(%(value)d of %(max)s)'),
                ' ', progressbar.Percentage(),
                ' ', progressbar.Bar()]
        self.dynamic_data = {k+'_'+kk: 0.0 for k in display_data.keys() for kk in display_data[k]}
        diff_data = {'diff_'+k+'_'+kk: 0.0 for k in display_data.keys() for kk in display_data[k]}
        self.dynamic_data.update(diff_data)
        for t in display_data.keys():
            ddstr = ' [' + t + ']'
            for s in display_data[t]:
                value_name = t + '_' + s
                ddstr = ddstr + ' ' + s + ':' + '%(' + value_name + ').3f (%(diff_' + value_name + ').3f)'
            self.widgets.append(progressbar.FormatLabel(ddstr))
        self.widgets.extend(['|', progressbar.FormatLabel('Time: %(elapsed)s'), '|', progressbar.AdaptiveETA()])


    def __call__(self, msg):
        # validation the input
        if isinstance(msg, dict):
            # train phase
            if msg['type'] == 'train':
                # create message
                vs = {'train_'+k: v for k, v in msg.items() if k in self.display_data['train']}
                dd = diff_from_history(vs, self.train_log_data, self.history_len)
                vs.update(dd)
                self.bar.update(msg['iteration'], **vs)

            # test phase
            elif msg['type'] == 'test':
                # create message
                vs = {'test_'+k: v for k, v in msg.items() if k in self.display_data['test']}
                dd = diff_from_history(vs, self.test_log_data, self.history_len)
                vs.update(dd)
                self.bar.update(**vs)


    def post_log(self):
        self.bar.finish()


    def run(self):
        # queue check
        assert self.queue is not None, \
            "Log Queue is None, use Logger.setQueue(queue) before calling me."

        self.stop = threading.Event()
        self.bar = MasalachaiProgressBar(
                max_value=self.max_value, 
                widgets=self.widgets,
                dynamic_data=self.dynamic_data)

        self.bar.start()

        while not self.stop.is_set():
            res = self.queue.get()
            if getattr(res, '__hash__', False) and res in self.mode:
                log_func = self.mode[res]
                if res == 'END':
                    self.stop.set()
                continue
            self.__call__(log_func(res))

        self.post_log()


    def set_max_value(self, value):
        self.max_value = value


    def log_train_progress(self, res):
        res['type'] = 'train'
        log_dict = {k: res[k] for k in res.keys()}
        return log_dict


    def log_test_progress(self, res):
        res['type'] = 'test'
        log_dict = {k: res[k] for k in res.keys()}
        return log_dict

