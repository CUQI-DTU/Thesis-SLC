################################################
# This script produces figures for thesis chapter 5
# related to the case study of CT of subsea pipes
# By Silja L. Christensen
# May 2024
################################################

#%%
import numpy as np
import matplotlib.pyplot as plt

import funs

from cil.framework import AcquisitionGeometry, ImageGeometry, AcquisitionData, BlockDataContainer
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.operators import BlockOperator, IdentityOperator, GradientOperator
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares, ZeroFunction
from cil.plugins.ccpi_regularisation.functions import FGP_TV


#%%
#=======================================================================
# Initialize
#=========================================================================

# Data: can be downloaded via https://zenodo.org/records/6817690
datapath = '../FORCE/data/Data_20180911/'
datafile = 'sinoN8.dat'

# Filepath for saving figures
path = 'figures/realrecon_geom{}_'.format(ag)

# Aqcuisition geometry type
# Choose "full", "limited90" or "sparseangles20percent"
ag = "limited90" 

# Reconstruction domain discretization size
N = 500

# Noise std
data_std = 0.05

# Colormap for plotting
cmap = 'gray'
cmin = -0.05
cmax = 0.2

# Projection backend
device = 'gpu'

# Optimization parameters
if ag == "full":
    noreg_iter = 100
    alpha_1st_tikh = 0.8
    tikh_1st_iter = 200 
    alpha_tikh = 3
    tikh_iter = 100
    alpha_TV = 0.1
    TV_iter = 500
elif ag == "limited90":
    noreg_iter = 100 
    alpha_1st_tikh = 0.8
    tikh_1st_iter = 200 
    alpha_tikh = 3 
    tikh_iter = 100 
    alpha_TV = 0.1
    TV_iter = 230
elif ag == "sparseangles20percent":
    noreg_iter = 100
    alpha_1st_tikh = 0.8
    tikh_1st_iter = 200 
    alpha_tikh = 3 
    tikh_iter = 100 
    alpha_TV = 0.1
    TV_iter = 230

#%%=======================================================================
# Define geometry
#=========================================================================
# Load aqusition geometry from library
det_count, angles, source_y, detector_y, beamshift_x, _, det_elem_size, det_length = funs.geom_Data20180911(ag)
angle_count = len(angles)

# Size of reconstruction domain
domain=(55, 55)

# Start view angle
if ag == "limited90":
    start_angle = -np.pi/2-2*15/180*np.pi
elif ag == "sparseangles20percent":
    start_angle = 4/180*np.pi
else:
    start_angle = 0

# CIL geometries
cil_ag = AcquisitionGeometry.create_Cone2D(source_position = [beamshift_x,-source_y], 
                                        detector_position = [beamshift_x,detector_y], 
                                        detector_direction_x=[1, 0], 
                                        rotation_axis_position=[0, 0])
cil_ag.set_angles(angles=angles, initial_angle = start_angle, angle_unit='radian')
cil_ag.set_panel(num_pixels=det_count, pixel_size=det_elem_size)

cil_ig = ImageGeometry(voxel_num_x=N, 
                   voxel_num_y=N, 
                   voxel_size_x=domain[0]/N, 
                   voxel_size_y=domain[0]/N)


#%%=======================================================================
# Load sinogram
#=========================================================================

# load data and change to correct data structure
sino = np.loadtxt(datapath + datafile, delimiter=';')
sino = sino.astype('float32')
sino_astra = np.rot90(sino, k = 1)
sino_astra = sino_astra[:-8, :]
if ag == 'sparseangles50percent':
    sino_astra  = sino_astra[::4, :]
elif ag == 'sparseangles20percent':
    sino_astra  = sino_astra[::10, :]
elif ag == 'sparseangles':
    sino_astra  = sino_astra[::20, :]
elif ag == 'full':
    sino_astra  = sino_astra[::2, :]
elif ag == 'limited90':
    sino_astra  = sino_astra[::2, :][15:105,:]
b_data = sino_astra.flatten()
b_data = b_data.reshape(sino_astra.shape).flatten(order="F")
data = AcquisitionData(array=np.rot90(b_data.astype(np.float32).reshape(det_count, angle_count)), geometry=cil_ag)

#%%=======================================================================
# Forward projector
#=========================================================================
# cil
A_cil = ProjectionOperator(cil_ig, cil_ag, device = device)

#%%=======================================================================
# CGLS reconstruction
#=========================================================================

F = LeastSquares(A_cil, data)
initial = cil_ig.allocate(0.0)
FISTA_noreg = FISTA(f = F, g = ZeroFunction(), initial=initial, max_iteration = 1000, update_objective_interval = 1)
FISTA_noreg.run(noreg_iter, verbose=True)
recon_noreg = FISTA_noreg.solution

# Figure 5.3, column 1, row 1/2/3 (ag="full"/"sparseangles20percent"/"limited90")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
cs = ax.imshow(np.rot90(recon_noreg.as_array(), k = 3), extent=[0, N, N, 0], aspect='equal', cmap=cmap, vmin = cmin, vmax = cmax)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.tick_params(axis='both', which='both', length = 0, labelsize=12)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
cbar.ax.tick_params(labelsize=12) 
plt.savefig(path + 'noreg.png', transparent=False)
plt.close()

#%%=======================================================================
# 1st order Tikhonov reconstruction
#=========================================================================

F = LeastSquares(A_cil, data)
L = GradientOperator(cil_ig)
operator_block = BlockOperator(A_cil, alpha_1st_tikh*L)
zero_data = L.range.allocate(0)
data_block = BlockDataContainer(data, zero_data)
F_T = LeastSquares(operator_block, data_block)
#setup CGLS with the Block Operator and Block DataContainer
initial = cil_ig.allocate(0.0)      
FISTA_1st_tikh = FISTA(f = F_T, g = ZeroFunction(), initial=initial, max_iteration = 1000, update_objective_interval = 1)
#run the algorithm
FISTA_1st_tikh.run(tikh_1st_iter, verbose = True)
recon_1st_Tik = FISTA_1st_tikh.solution

# Figure 5.3, column 2, row 1/2/3 (ag="full"/"sparseangles20percent"/"limited90")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
cs = ax.imshow(np.rot90(recon_1st_Tik.as_array(), k = 3), extent=[0, N, N, 0], aspect='equal', cmap=cmap, vmin = cmin, vmax = cmax)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.tick_params(axis='both', which='both', length = 0, labelsize=12)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
cbar.ax.tick_params(labelsize=12) 
plt.savefig(path + '1st_Tik.png', transparent=False)
plt.close()

#%%=======================================================================
# TV reconstruction
#=========================================================================

F = LeastSquares(A_cil, data)
TV = FGP_TV(alpha=alpha_TV, device='gpu')
initial = cil_ig.allocate(0.0)
my_TV = FISTA(f=F, 
                g=TV,
                initial=initial, 
                max_iteration=2000, 
                update_objective_interval=1)
my_TV.run(TV_iter, verbose=True)
recon_TV = my_TV.solution

# Figure 5.3, column 3, row 1/2/3 (ag="full"/"sparseangles20percent"/"limited90")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
cs = ax.imshow(np.rot90(recon_TV.as_array(), k = 3), extent=[0, N, N, 0], aspect='equal', cmap=cmap, vmin = cmin, vmax = cmax)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.tick_params(axis='both', which='both', length = 0, labelsize=12)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
cbar.ax.tick_params(labelsize=12) 
plt.savefig(path + 'TV.png', transparent=False)
plt.close()

#%%=======================================================================
# Plot sinogram
#=========================================================================

# Figure 5.1(b)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4))
cs = ax.imshow(b_data.reshape(det_count,angle_count, order = 'C'), cmap=cmap, aspect=0.5)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax) 
ax.tick_params(axis='both', which='both', labelsize=12)
cbar.ax.tick_params(labelsize=12) 
ax.set_xlabel('Projection angle [deg]', fontsize = 12)
ax.set_ylabel('Detector pixel', fontsize = 12)
ax.set_xticks(np.linspace(0,angle_count,7, endpoint=True))
plt.savefig(path + 'sinogram.png', transparent=False)
plt.close()


