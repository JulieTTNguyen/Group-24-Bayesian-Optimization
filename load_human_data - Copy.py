#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to load and reproduce human data
@author: khma
"""

import numpy as np
import numba

data = np.load('game_results_humans.npz')
xyz = data['xyz']
params = data['params']
meta = data['meta']

# Ground truth model in numpy
def gmm_model(params, x, y):
    # params: (N, 25), x/y: (N, 20)
    g = params[:24].reshape(6, 4)
    w, mx, my, s = g[:, 0], g[:,1], g[:, 2], g[:, 3]
    expo = -((x[...,None] - mx[None])**2 + (y[...,None] - my[None])**2) / s[None]
    res = np.sum(w[None] * np.exp(expo), axis=-1)
    
    return res * params[24]


for i in range(len(params)):
    assert np.allclose(gmm_model(params[i], xyz[i,:,0], xyz[i,:,1]), xyz[i,:,2])
    
    
import matplotlib.pyplot as plt

# Plot the ground truth and data for the first game

plt.figure()
plt.subplot(1,2,1)
x = np.linspace(0,1,100)
grid = np.meshgrid(x,x)
zgrid = gmm_model(params[0],grid[0],grid[1])
plt.imshow(zgrid,extent=[0, 1, 0, 1])
plt.colorbar()
plt.title(f'Player: {meta[0,1]}, game ID: {meta[0,0]}')
plt.plot(xyz[0,:,0],xyz[0,:,1],'kx-')
best_i = np.argmax(xyz[0,:,2])
plt.plot(xyz[0,best_i,0],xyz[0,best_i,1],'ro')


#Plot mean curve for best performance, for all players, 4 first are fixed seeds
plt.subplot(1,2,2)
plt.plot(np.maximum.accumulate(xyz[:,:,2],axis=1).mean(0))
plt.title('Mean evolution over trials')
