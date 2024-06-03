###########################################
# This script produces figures for thesis chapter 2
# related to the optimization approach to CT
# Silja L. Christensen
# May 2024
# Based on CIL demos: 
#      https://github.com/TomographicImaging/CIL-Demos/blob/main/demos/1_Introduction/04_FBP_CGLS_SIRT.ipynb
###########################################

import numpy as np
import matplotlib.pyplot as plt
# cil imports
from cil.framework import ImageGeometry, BlockDataContainer
from cil.framework import AcquisitionGeometry
from cil.optimisation.algorithms import CGLS, FISTA
from cil.optimisation.operators import BlockOperator, IdentityOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import LeastSquares
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.plugins import TomoPhantom
from cil.utilities.display import show2D

#%% Setup test problem

# Backend for FBP and the ProjectionOperator
device = 'gpu'
# Colormap for plotting
cmap = "viridis"
# Pixel discretization size in reconstruction
n_pixels = 256
# View angles
angles = np.linspace(0, 180, 50, endpoint=False, dtype=np.float32) # 256

# CIL acquisition geometry
ag = AcquisitionGeometry.create_Parallel2D()\
                            .set_angles(angles)\
                            .set_panel(n_pixels, pixel_size=1/n_pixels)

# CIL image geometry
ig = ImageGeometry(voxel_num_x=n_pixels, 
                   voxel_num_y=n_pixels, 
                   voxel_size_x=1/n_pixels, 
                   voxel_size_y=1/n_pixels)

# Get Shepp-Logan phantom
phantom = TomoPhantom.get_ImageData(num_model=1, geometry=ig)

# Create projection operator using Astra-Toolbox.
A = ProjectionOperator(ig, ag, device)

# Create an acquisition data (numerically)
sino = A.direct(phantom)

# Add noise
# Incident intensity: lower counts will increase the noise
background_counts = 50000 
# Convert the simulated absorption sinogram to transmission values using Lambert-Beer. 
# Use as mean for Poisson data generation.
# Convert back to absorption sinogram.
counts = background_counts * np.exp(-sino.as_array())
tmp = np.exp(-sino.as_array())
noisy_counts = np.random.poisson(counts)
nonzero = noisy_counts > 0
sino_out = np.zeros_like(sino.as_array())
sino_out[nonzero] = -np.log(noisy_counts[nonzero] / background_counts)
# Allocate sino_noisy and fill with noisy data
sino_noisy = ag.allocate()
sino_noisy.fill(sino_out)

#%% FBP reconstruction
fbp = FBP(ig, ag, device)
recon_fbp = fbp(sino_noisy)

#%% Tikhonov
alpha = 0.02
L = IdentityOperator(ig)
operator_block = BlockOperator(A, alpha*L)
zero_data = L.range.allocate(0)
data_block = BlockDataContainer(sino_noisy, zero_data)
#setup CGLS with the Block Operator and Block DataContainer
initial = ig.allocate(0)      
cgls_tikh = CGLS(initial=initial, operator=operator_block, data=data_block, update_objective_interval = 10)
cgls_tikh.max_iteration = 1000
#run the algorithm
cgls_tikh.run(200)

#%% TV
alpha_TV = 0.001
F = LeastSquares(A, sino_noisy)
GTV = alpha_TV*FGP_TV(device='gpu') 
initial = ig.allocate(0.0)
my_TV = FISTA(f=F, 
                g=GTV,
                initial=initial, 
                max_iteration=2000, 
                update_objective_interval=10)
my_TV.run(2000, verbose=True) 

#%% Plotting

solutions = [phantom, recon_fbp, cgls_tikh.solution, my_TV.solution]
titles = ['Phantom', 'FBP', 'Tikhonov', 'TV']
cmin = -0.2
cmax = 1.2

# Figure 2.2: Deterministic Shepp-Logan reconstructions
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (5,5), gridspec_kw = {'wspace': 0.2, 'hspace': 0.2})
ax = ax.flatten()
for i in range(4):
       cs = ax[i].imshow(solutions[i].as_array(), vmin = cmin, vmax = cmax, aspect='equal', cmap=cmap)
       ax[i].tick_params(axis='both', which='both', length=0)
       plt.setp(ax[i].get_xticklabels(), visible=False)
       plt.setp(ax[i].get_yticklabels(), visible=False)
       ax[i].set_title(titles[i], fontsize = 12)
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([ax[3].get_position().x1+0.03,ax[3].get_position().y0,0.03,ax[1].get_position().y1-ax[3].get_position().y0])
cbar = plt.colorbar(cs, cax=cax) 
cbar.set_ticks(np.linspace(cmin, cmax, 8, endpoint = True))
plt.savefig("regularization.png")