################################################
# This script produces figures for thesis chapter 3
# related to the Bayesian linear regression example
# By Silja L. Christensen
# May 2024
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from cuqi.array import CUQIarray
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.model import LinearModel
from cuqi.geometry import Discrete, Continuous1D
from cuqi.sampler import MetropolisHastings, Linear_RTO

#%% IACT function
def iact(dati):
    # Implementation from Computational Uncertainty Quantification for Inverse Problems by John Bardsley
        
    if len(np.shape(dati)) == 1:
        dati =  dati[:, np.newaxis]

    mx, nx  = np.shape(dati)
    
    
    tau = np.zeros(nx)
    m   = np.zeros(nx)
    
    x       = np.fft.fft(dati, axis=0)
    xr      = np.real(x)
    xi      = np.imag(x)
    xr      = xr**2 + xi**2
    xr[0,:] = 0
    xr      = np.real(np.fft.fft(xr, axis=0))
    var     = xr[0,:] / len(dati) / (len(dati)-1)
    
    for j in range(nx):
        if var[j] == 0:
            continue
        
        xr[:,j] = xr[:,j]/xr[0,j]
        summ    = -1/3
        
        for i in range(len(dati)):
            summ = summ + xr[i,j] - 1/6
            if summ < 0:
                tau[j]  = 2*(summ + (i-1)/6)
                m[j]    = i
                break
                
    return tau, m

#%% Test problem
# Dimensions
n = 2
m = 5
# Unknown parameter
x_truevec = np.array([3,2])
x_true = CUQIarray(x_truevec, geometry=Discrete(['x1', 'x2']))
# Observation times
t = np.linspace(0,m,m, endpoint=False)
# Forward model
Amat = np.ones((m,n))
Amat[:,1] = t

#%% CUQI test problem

# Model
A = LinearModel(Amat)
A.domain_geometry = Discrete(n)
A.range_geometry = Continuous1D(t)

# Prior
x_mean = np.zeros(n)
x_std = np.ones(n)
x = Gaussian(x_mean, sqrtcov = x_std, geometry = A.domain_geometry)

# Likelihood
y_std = np.ones(m)
y = Gaussian(A(x), sqrtcov=y_std, geometry = A.range_geometry)

# True and observed noisy data
y_true = A(x_true)
np.random.seed(10)
y_obs = y(x=x_true).sample()

# Posterior
posterior = JointDistribution(x, y)(y=y_obs)

# Figure 3.1: Exact and noisy data
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
ax.plot(t, y_obs, 'o', color = "tab:blue", label="Noisy data")
ax.plot(t, y_true, linestyle = '-', color = "tab:red", label="Exact data")
ax.set_xlabel('t')
ax.set_ylabel('y')
plt.legend()
plt.savefig("regression_data.png")

#%% Deterministic least squares fit
x_determ = np.linalg.lstsq(Amat, y_obs)[0]

#%% Analytical posterior by conjugacy
prec_conjugate = Amat.T@Amat+np.eye(n)
cov_conjugate = np.linalg.inv(prec_conjugate)
mean_conjugate = (Amat.T@y_obs)@cov_conjugate

print('Analytical posterior by conjugacy: ')
print("mean conjugate: {}".format(mean_conjugate))
print("cov conjugate: {}".format(cov_conjugate))

# Figure 3.2(a): Contour plot of likelihood
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
data_tmp = np.vstack((data[:,:,0].flatten(),data[:,:,1].flatten()))
z_like = (1/(2*np.pi))**(m/2)*np.exp(-1/2*np.linalg.norm(Amat@data_tmp -y_obs[:, np.newaxis], axis = 0)**2)
z_like = np.reshape(z_like, xx.shape)
cs = axes[0,0].contourf(xx, yy, z_like, levels = 16)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_analyticallike.png")

# Figure 3.2(b): Contour plot of prior
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
xx, yy = np.mgrid[-5:5.1:.1, -5:5.1:.1]
data = np.dstack((xx, yy))
z_prior = 1/(2*np.pi)*np.exp(-1/2*np.linalg.norm(data, axis = 2)**2)
cs = axes[0,0].contourf(xx, yy, z_prior, levels = 16)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_analyticalprior.png")

# Figure 3.2(c): Contour plot of prior-likelihood product
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].contourf(xx, yy, z_like * z_prior, levels = 16)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_analyticalpriortimeslike.png")

# Figure 3.2(d): Contour plot of analytical posterior 
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
xx, yy = np.mgrid[-5:5.1:.1, -5:5.1:.1]
rv = multivariate_normal(mean_conjugate, cov_conjugate)
data = np.dstack((xx, yy))
z = rv.pdf(data)
cs = axes[0,0].contourf(xx, yy, z, levels = 16)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_analyticalpost.png")

# Figure 3.3(a): Contour plot of analytical posterior, closer zoom
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
xx, yy = np.mgrid[0:5.1:.1, 1:3.1:.1]
rv = multivariate_normal(mean_conjugate, cov_conjugate)
data = np.dstack((xx, yy))
z = rv.pdf(data)
cs = axes[0,0].contourf(xx, yy, z, levels = 16)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_contours.png")

#%% Sample posterior with MH
np.random.seed(1000000)
MHsampler = MetropolisHastings(posterior, x0 = np.array([10,10]), scale=0.5)
samplesMH = MHsampler.sample(10000, 0)
samplesMH_burnin = samplesMH.burnthin(100)

meanMH_burnin = samplesMH_burnin.mean()
varMH_burnin = samplesMH_burnin.variance()
meanMH_burnthin = samplesMH.burnthin(Nb = 100, Nt = 20).mean()
varMH_burnthin = samplesMH.burnthin(Nb = 100, Nt = 20).variance()

print("MH burnin mean: {}".format(meanMH_burnin))
print("MH burnthin mean: {}".format(meanMH_burnthin))
print("MH burnin var: {}".format(varMH_burnin))
print("MH burnthin var: {}".format(varMH_burnthin))

# Figure 3.3(b): 2D histogram of MH samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].hist2d(samplesMH_burnin.samples[0,:], samplesMH_burnin.samples[1,:], bins=[50,30], range=[[0, 5], [1, 3]], density=False)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs[3], cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_MHhist2d.png")

# #%% Sample posterior with linear RTO
np.random.seed(1000000)
RTOsampler = Linear_RTO(posterior, x0 = np.array([10,10]))
samplesRTO = RTOsampler.sample(10000, 0)
samplesRTO_burnin = samplesRTO.burnthin(100)

meanRTO = samplesRTO_burnin.mean()
varRTO = samplesRTO_burnin.variance()

# Figure 3.3(c): 2D histogram of RTO samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].hist2d(samplesRTO_burnin.samples[0,:], samplesRTO_burnin.samples[1,:], bins=[50,30], range=[[0, 5], [1, 3]], density=False)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs[3], cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_RTOhist2d.png")

#%% Sample posterior with conjugate
np.random.seed(1000000)
samplesconjugate = Gaussian(mean = mean_conjugate, cov = cov_conjugate).sample(10000)
samplesconjugate_burnin = samplesconjugate.burnthin(100)

meanconjugatesample = samplesconjugate_burnin.mean()
varconjugatesample = samplesconjugate_burnin.variance()

# Figure 3.3(d): 2D histogram of direct/conjugate samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = axes[0,0].hist2d(samplesconjugate_burnin.samples[0,:], samplesconjugate_burnin.samples[1,:], bins=[50,30], range=[[0, 5], [1, 3]], density=False)
axes[0,0].plot(x_true[0], x_true[1], 'r*', markersize = 10)
axes[0,0].set_xlabel('x1', fontsize=12)
axes[0,0].set_ylabel('x2', fontsize=12)
fig.subplots_adjust(left = 0.15, right=0.85)
cax = fig.add_axes([axes[0,0].get_position().x1+0.01,axes[0,0].get_position().y0,0.03,axes[0,0].get_position().height])
cbar = plt.colorbar(cs[3], cax=cax) 
axes[0,0].tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12)
plt.savefig("regression_conjugatesamplehist2d.png")

#%% Posterior diagnostics
iactMH, _ = iact(samplesMH_burnin.samples[0,:])
print("MH x1 iact: {}".format(iactMH))
essMH = samplesMH_burnin.compute_ess()
print("MH ess: {}".format(essMH))

# Figure 3.5(a): Chain with burnin phase, MH x1
fig = plt.figure(figsize=(5,2))
plt.plot(samplesMH.samples[0,:1000], label = "Chain")
plt.axvline(x=100, label = 'End of burn-in', color = "tab:orange")
plt.ylim([0, 12])
plt.xlabel("Realization number")
plt.ylabel("Parameter value")
plt.legend()
plt.tight_layout() 
plt.savefig("regression_burnin_x1.png")

# Figure 3.5(c): Chain with burnin phase, MH x2
fig = plt.figure(figsize=(5,2))
plt.plot(samplesMH.samples[1,:1000], label = "Chain")
plt.axvline(x=100, label = 'End of burn-in', color = "tab:orange")
plt.ylim([0, 12])
plt.xlabel("Realization number")
plt.ylabel("Parameter value")
plt.legend()
plt.tight_layout() 
plt.savefig("regression_burnin_x2.png")

# Figure 3.5(b): Autocorrelation function, MH x1
samplesMH_burnin.plot_autocorrelation(variable_indices=[0], figsize=(5,2))
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title(' ')
plt.tick_params(axis='both', labelsize=10)
plt.savefig("regression_autocorrelation_x1.png")

# Figure 3.5(d): Autocorrelation function, MH x2
samplesMH_burnin.plot_autocorrelation(variable_indices=[1], figsize=(5,2))
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.title(' ')
plt.tick_params(axis='both', labelsize=10)
plt.savefig("regression_autocorrelation_x2.png")

#%% Posterior visualization
# Figure 3.6: 1D and 2D marginals of MH samples
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,4), squeeze=False)
cs = samplesMH_burnin.plot_pair(marginals = True, marginal_kwargs = {'kind': 'hist', 'hist_kwargs': {'bins': 30}})
cs[1,0].set_xlabel('x1', fontsize=12)
cs[1,0].set_xticks(np.linspace(0,5,6, endpoint= True))
cs[1,0].set_yticks(np.linspace(1,3,5, endpoint= True))
cs[1,0].set_ylim([1.2, 3.2])
cs[1,0].set_ylabel('x2', fontsize=12)
cs[1,0].tick_params(axis='both', labelsize=12)
plt.savefig("regression_pairplot.png")

# Figure 3.7(a): Mean and credible interval of MH samples in parameter space
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
lo_conf, up_conf = samplesMH_burnin.compute_ci(95)
ax.errorbar(('x1', 'x2'), lo_conf,
                            yerr=np.vstack(
                                (np.zeros(len(lo_conf)), up_conf-lo_conf)),
                            color = 'dodgerblue', fmt = 'none' , capsize = 3, capthick = 1, label = "Probabilistic 95% CI")
ax.plot(('x1', 'x2'), x_truevec, 'o', color = 'tab:red', label="Truth")
ax.plot(('x1', 'x2'), x_determ, 'v', color = 'tab:orange', label = "Deterministic fit")
ax.plot(('x1', 'x2'), samplesMH_burnin.mean(), 's', color = 'tab:blue', label = "Probabilistic mean")
ax.set_xlabel('parameter')
ax.set_ylabel('value')
plt.legend()
plt.savefig("regression_mean_modelspace.png")

# Figure 3.7(b): Mean and credible interval of MH samples in data space
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
lo_conf, up_conf = A(samplesMH_burnin).compute_ci(95)
ax.fill_between(t,up_conf, lo_conf, label = "Probabilistic 95% CI", color = 'dodgerblue', alpha = 0.25)
ax.plot(t, A(samplesMH_burnin).mean(), linestyle = '--', label = "Probabilistic mean")
ax.plot(t, A(x_determ), ':', label = 'Deterministic fit')
ax.plot(t, y_obs, 'o', label="Noisy data")
ax.plot(t, y_true, linestyle = '-', label="Exact data")
ax.set_xlabel('t')
ax.set_ylabel('y')
plt.legend()
plt.savefig("regression_mean_dataspace.png")

