#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import skimage
from skimage import io, feature, filters, color
from scipy.signal import convolve2d
from scipy.ndimage import filters

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


def gaussian_gradient_magnitude(im, std):
    def kernel_size(s):
        return 2 * math.ceil(2 * s) + 1
    def kernel_x(x, y, s):
        return -(x / (2 * math.pi * s**4)) * np.exp(-(x**2 + y**2) / (2 * s**2))
    def kernel_y(x, y, s):
        return -(y / (2 * math.pi * s**4)) * np.exp(-(x**2 + y**2) / (2 * s**2))
    # gaussian gradient kernel
    k = kernel_size(std)
    gx, gy = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kx = kernel_x(gx, gy, std)
    ky = kernel_y(gx, gy, std)
    # get gradient
    grad_x = convolve2d(im, kx, 'same')
    grad_y = convolve2d(im, ky, 'same')
    # get magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

def joint_bilateral_filter(sdm, im, spatial_domain, sr):
    wr = int(math.ceil(spatial_domain * 4) + 1)
    padding = (wr - 1) / 2
    padn = ((padding, padding), (padding, padding), (0, 0))
    padded_im = np.pad(im, padn, mode='edge')

    if sdm.ndim==2:
        sdm = np.expand_dims(sdm, 2)
    padded_sdm = np.pad(sdm, padn, mode='edge')
    patch_sdm = rolling_window(padded_sdm, (wr, wr))
    patch_im = rolling_window(padded_im, (wr, wr))

    p = ((0,0),(0,0),(0,0),(wr-1,0),(wr-1,0))
    #tmp = np.exp(-(patch_im - np.pad(im[:,:,:,np.newaxis,np.newaxis], p, mode='reflect'))**2 / (2 * sr**2)).sum()
    tmp = np.pad(im[:,:,:,np.newaxis,np.newaxis], p, mode='reflect')

# load image as grey
fname = './lena.tiff'
#fname = './input.png'
im = skimage.img_as_float(io.imread(fname))
im_grey = color.rgb2grey(im)

# canny edge detection
sigma = 1.
edge_map = feature.canny(im_grey, sigma)

# estimate defocus map
std_1 = 1.
std_2 = std_1 * 2.5
ratio = gaussian_gradient_magnitude(im_grey, std_1) / gaussian_gradient_magnitude(im_grey, std_2)

# get edge index
sdm = np.zeros_like(ratio)
mx, my = np.where(edge_map * (ratio > 1.01) * (ratio <= std_2/std_1))

# enhancement
sdm[mx,my] = np.sqrt((ratio[mx,my]**2 * (std_1**2 - std_2**2) + 0.001) / (1 - ratio[mx,my]**2))

# clip
max_blur = 3
sdm = np.where(sdm > max_blur, max_blur, sdm)

# joint bilateral filtering
joint_bilateral_filter(sdm, im, 5, 0.1*max_blur)

#io.imsave('test.png', sdm.astype(np.uint8)*15)
#io.imsave('test.png', 255*edge_map.astype(np.uint8))
