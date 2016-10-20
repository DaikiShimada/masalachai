#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import six  
from six.moves.urllib import request

import struct
import gzip

train_image = 'train-images-idx3-ubyte.gz'
train_label = 'train-labels-idx1-ubyte.gz'
test_image = 't10k-images-idx3-ubyte.gz'
test_label = 't10k-labels-idx1-ubyte.gz'

def download():
    url = 'http://yann.lecun.com/exdb/mnist'

    request.urlretrieve(url+'/'+train_image, train_image)
    request.urlretrieve(url+'/'+train_label, train_label)
    request.urlretrieve(url+'/'+test_image, test_image)
    request.urlretrieve(url+'/'+test_label, test_label)

def convert_image(fname):
    with gzip.open(fname, 'rb') as f:
        # unpack as big endian unsigned int, unsigned int, unsigned int, unsigned int
        magic, n, r, c = struct.unpack('>IIII', f.read(16))
        images = np.array([ord(f.read(1)) for i in six.moves.range(n*r*c)], dtype=np.uint8).reshape(n, r, c)
        return images

def convert_label(fname):
    with gzip.open(fname, 'rb') as f:
        # unpack as big endian unsigned int, unsigned int
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.array([ord(f.read(1)) for i in six.moves.range(n)], dtype=np.uint8)
        return labels

def load(name='mnist.pkl'):
    with open(name, 'rb') as data:
        mnist = six.moves.cPickle.load(data)
    return mnist


if __name__ == '__main__':
    #download()
    train_image_ary = convert_image(train_image)
    test_image_ary = convert_image(test_image)
    train_label_ary = convert_label(train_label)
    test_label_ary = convert_label(test_label)

    category_names = [str(i) for i in six.moves.range(10)]

    train = {'data': train_image_ary, 'target': train_label_ary, 'size': len(train_label_ary), 'categories': len(category_names), 'category_names': category_names}
    test = {'data': test_image_ary, 'target': test_label_ary, 'size': len(test_label_ary), 'categories': len(category_names), 'category_names': category_names}
    data = {'train': train, 'test': test}

    out_name = 'mnist.pkl'
    with open(out_name, 'wb') as out_data:
        six.moves.cPickle.dump(data, out_data, -1)
