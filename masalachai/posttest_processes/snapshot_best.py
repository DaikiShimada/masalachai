# -*- coding: utf-8 -*-

import os
from masalachai.posttest_process import PostTestProcess
from chainer import serializers

class SnapshotBest(PostTestProcess):

    def __init__(self, model, filename, observed='loss', patience=1):
        self.model = model
        self.filename = filename
        self.observed = observed
        self.patience = patience
        self.iteration = 0
        self.loss = None

    def save_model(self, test_res):
        # Adding contents of test_res to instance variables, 
        # we can use them in filename format.
        self.__dict__.update({k:v for k,v in test_res.items() if k not in self.__dict__})
        filename = self.filename.format(self.__dict__)
        serializers.save_npz(filename, self.model)

    def __call__(self, test_res):
        value = test_res[self.observed]
        self.iteration = test_res['iteration']

        if self.iteration > self.patience:
            if self.loss is None or value < self.loss:
                self.loss = value
                self.save_model(test_res)
        return False