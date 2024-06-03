################################################
# This script produces figures for thesis chapter 4
# related to the Bayesian approach to CT
# By Silja L. Christensen
# May 2024
################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plot

import cuqi
import cuqipy_cil
import cil 

from cuqipy_cil.testproblem import ParallelBeam2D
from cuqi.distribution import Gaussian, GMRF, JointDistribution, Laplace_diff, InverseGamma
from cuqi.sampler import Linear_RTO, UnadjustedLaplaceApproximation, Gibbs

#%% Settings

# CT backend
cuqipy_cil.config.PROJECTION_BACKEND = "astra"
cuqipy_cil.config.PROJECTION_BACKEND_DEVICE = "gpu"

# Colormap for plotting
cmap = "viridis"

# Sinogram noise std
noise_std = 0.5

# Type of prior: "GMRF" produces figure 4.1, "LMRF" produces figure 4.2
prior = "LMRF"

# Reconstruction domain discretization size
N = 256

# Number of detectors in panel
num_det = 256

# View angles
num_angles = 50
angles = np.linspace(0, 180, num_angles, endpoint=False, dtype=np.float32) # 256

#%% Bayesian problem
# CT test problem from CUQIpy-CIL
TP = ParallelBeam2D(im_size=(N, N),
                        det_count=num_det,
                        angles=np.linspace(0, np.pi, num_angles),
                        phantom = "shepp-logan",
                        noise_type="gaussian",
                        noise_std=noise_std)
# CT model
A = TP.model
# Data observation
y_obs = TP.data

# Data distribution
y = Gaussian(mean=A, sqrtcov=noise_std, geometry = A.range_geometry)

# Prior
if prior == "GMRF":
    x = GMRF(mean=np.zeros(A.domain_dim), prec=1e2, physical_dim=2, geometry=A.domain_geometry)
elif prior == "LMRF":
    x = Laplace_diff(location=np.zeros(A.domain_dim), scale = 5e-2, physical_dim=2, geometry=A.domain_geometry)

# Posterior
posterior = JointDistribution(y, x)(y = y_obs)

# Sampler
if prior == "GMRF":
    sampler = Linear_RTO(posterior)
elif prior == "LMRF":
    sampler = UnadjustedLaplaceApproximation(posterior)

#%% Run sampler
np.random.seed(1000)
samples = sampler.sample(1000,100)

#%% Plotting

# Figure 4.1(a)/4.2(a): Plot x mean
fig, ax = plt.subplots(1,1, figsize=(4,3))
cs = samples.plot_mean(cmap = cmap)
cs[0].axes.tick_params(axis='both', which='both', length=0)
plt.setp(cs[0].axes.get_xticklabels(), visible=False)
plt.setp(cs[0].axes.get_yticklabels(), visible=False)
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
cbar = plt.colorbar(cs[0], cax=cax)
plt.savefig("CUQI_{}mean.png".format(prior))

# Figure 4.1(b)/4.2(b): Plot x std
fig, ax = plt.subplots(1,1, figsize=(4,3))
cs = samples.plot_std(cmap = cmap)
cs[0].axes.tick_params(axis='both', which='both', length=0)
plt.setp(cs[0].axes.get_xticklabels(), visible=False)
plt.setp(cs[0].axes.get_yticklabels(), visible=False)
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([cs[0].axes.get_position().x1+0.01,cs[0].axes.get_position().y0,0.03,cs[0].axes.get_position().height])
cbar = plt.colorbar(cs[0], cax=cax)
plt.savefig("CUQI_{}std.png".format(prior))