# -*- coding: utf-8 -*-

import numpy as np

def rolling_window_lastaxis(ary, w):
    shape = ary.shape[:-1] + (ary.shape[-1] - w + 1, w)
    strides = ary.strides + (ary.strides[-1],)
    return np.lib.stride_tricks.as_strided(ary, shape=shape, strides=strides)

def rolling_window(ary, w):
    ''' 
    from http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter
    '''
    if not hasattr(w, '__iter__'):
        return rolling_window_lastaxis(ary, w)
    for i, d in enumerate(w):
        if d > 1:
            ary = ary.swapaxes(i, -1)
            ary = rolling_window_lastaxis(ary, d)
            ary = ary.swapaxes(-2, i)
    return ary
