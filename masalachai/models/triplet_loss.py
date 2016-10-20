# -*- coding: utf-8 -*-

import numpy
from chainer import cuda
from chainer import function


class TripletLoss(function.Function):
    
    """Triplet loss function."""
    
#    def __init__(self):
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        
        anchor, positive, negative = inputs
        N = anchor.shape[0]

        def euclidean_d(v1, v2):
            return chainer.functions.sum(chainer.functions.math.sqrt((v1-v2) ** 2), axis=1)
        d_ap = euclidean_d(anchor, positive)
        d_an = euclidean_d(anchor, negative)

        dist_p = chainer.functions.exp(d_ap)\
                / (chainer.functions.exp(d_ap)\
                + chainer.functions.exp(d_an))
        loss = chainer.functions.sum(dist_p * dist_p) / N
        return xp.array(loss, dtype=numpy.float32),

def tripletloss(anchor, positive, negative):
    """Computes triplet loss.
    Args:
        anchor (~chainer.Variable): The anchor example variable. The shape should be :math: `(N, K)`, where :math: `N` denotes the minibath size, and :math:`K` denotes the dimension of the anchor.
        positive (~chainer.Variable): The positive example variable. The shape should be the same as anchor.
        negative (~chainer.Variable): The negative example variable. The shape should be the same as anchor.

    Returns:
        -chainer.Variable: A variable holding a scalar that is the loss value calculated by the above equation.
    .. note::
        This isn't different from chainer's official triplet loss function.
        This cost can be used to train triplet networks. See `Learning\
        "Deep Metric Learning Using Triplet Network"\
        <https://arxiv.org/abs/1412.6622>
    """
    return TripletLoss()(anchor, positive, negative)
