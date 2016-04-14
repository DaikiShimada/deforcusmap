#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import skimage
from skimage import io

import rollingwin

def matting_laplacian_matrix(im, r):
    print rollingwin.rolling_window(im, (3, 3)).shape

fname = './lena.tiff'
im = skimage.img_as_float(io.imread(fname))

r = 1
matting_laplacian_matrix(im, r)
