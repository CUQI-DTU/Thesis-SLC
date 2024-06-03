################################################
# By Silja L. Christensen
# April 2024

# This script generates the thesis front page image
################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import funs

# Filepath for saving image
path = 'frontpageimage/'
os.makedirs(path, exist_ok=True)

# Get phantom
phantomname = 'DeepSeaOilPipe8'   # choose phantom
n = 1024                        # phantom dimension
phantom, _, _, _, _ = getattr(funs, phantomname)(n,True)

# DTU colors
dtuyellow = np.array([0.9647,0.8157,0.3019])
dtunavyblue = np.array([0.0118,0.0588,0.3098])
dtured = np.array([0.6,0,0])
dtugrey = np.array([0.8549,0.8549,0.8549])
dtuwhite = np.array([1,1,1])
dtugreen = np.array([0,0.5333,0.2078])
dtupurple = np.array([0.4745,0.1373,0.5569])
dtublue = np.array([0.1843,0.2431,0.9176])
dtubrightgreen = np.array([0.1216,0.8157,0.5098])

# Custom colormap
Ncol = 256
vals = np.ones((Ncol, 4))
for i in range(3):
        vals[:, i] = np.linspace(dtured[i], dtuwhite[i], Ncol)
newcmp = ListedColormap(vals)

# Front page image
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(wspace=.5)

ax = plt.subplot(111)
imagevec = phantom.flatten(order='F')
cs = ax.imshow(np.rot90(imagevec.reshape((n,n), order = "C")), extent=[0, n, n, 0], aspect='equal', cmap=newcmp, vmin = 0, vmax = 0.16)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(cs, cax=cax)
ax.tick_params(axis='both', which='both', length=0)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.savefig(path + 'frontpageimage_dturedwhite.png', transparent=False, dpi = 500)
plt.close()

