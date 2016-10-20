# -*- coding: utf-8 -*-

from masalachai.model import Model
from chainer.functions import triplet

## Triplet Network Wrapper
class TripletModel(Model):
    def __init__(self, predictor, lossfun=triplet):
        super(TripletModel, self).__init__(predictor, lossfun)

    def __call__(self, x, t=None, train=True):
        self.y = None
        self.loss = None
        xa, xp, xn = x
        self.ya = self.predictor(xa, train=train)
        self.yp = self.predictor(xp, train=train)
        self.yn = self.predictor(xn, train=train)
        self.loss = self.lossfun(self.ya, self.yp, self.yn)
        return self.loss

    def predict(self, x):
        x0, = x
        self.y = self.predictor(x0, train=False)
        return self.y

