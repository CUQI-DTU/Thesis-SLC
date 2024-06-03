################################################
# This script produces figures for thesis chapter 3
# related to the Bayesian linear regression problem 
# with hierarchical prior
# By Silja L. Christensen
# May 2024
################################################

import numpy as np
import matplotlib.pyplot as plt
import cuqi

import sys, os

#%% Test problem
# Dimensions
n = 2
m = 5
# Unknown parameter
x_truevec = np.array([3,2])
x_true = cuqi.array.CUQIarray(x_truevec, geometry=cuqi.geometry.Discrete(['x1', 'x2']))
# Observation times
t = np.linspace(0,m,m, endpoint=False)
# Forward model
Amat = np.ones((m,n))
Amat[:,1] = t

#%% CUQI test problem
# Model
A = cuqi.model.LinearModel(Amat, domain_geometry=cuqi.geometry.Discrete(n), range_geometry=cuqi.geometry.Continuous1D(t))
# Hyperprior
d = cuqi.distribution.Gamma(1,1)
# Prior
x_mean = np.zeros(n)
x_std = np.ones(n)
x = cuqi.distribution.Gaussian(x_mean, cov = lambda d: 1/d)
# Likelihood
y_std = np.ones(m)
y = cuqi.distribution.Gaussian(A(x), sqrtcov=y_std)
# True and observed noisy data
y_true = A(x_true)
np.random.seed(10)
y_obs = y(x=x_true).sample()
# Posterior
posterior = cuqi.distribution.JointDistribution(d, x, y)(y=y_obs)

# Deterministic LSQ solution
x_determ = np.linalg.lstsq(Amat, y_obs)[0]

#%% RTO Sampling

np.random.seed(1000000)
sampling_strategy = {'d': cuqi.sampler.Conjugate, 'x': cuqi.sampler.Linear_RTO}
RTOsampler = cuqi.sampler.Gibbs(posterior, sampling_strategy)
samplesRTO = RTOsampler.sample(10000, 100)

# Remove burnin
samplesRTO_x_burnin = samplesRTO['x'].burnthin(100)
samplesRTO_d_burnin = samplesRTO['d'].burnthin(100)

# Figure 3.4(a): 2D histogram of x samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].hist2d(samplesRTO_x_burnin.samples[0,:], samplesRTO_x_burnin.samples[1,:], bins=[50,30], range=[[0, 5], [1, 3]], density=False)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs[3], cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_hyper_RTO_xhist2d.png")

# Figure 3.4(b): 1D histogram of delta samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].hist(samplesRTO_d_burnin.samples[0,:], bins = 50)
axes[0,0].set_xlabel(r'$\delta$', fontsize=12)
plt.savefig("regression_hyper_RTO_dhist.png")
