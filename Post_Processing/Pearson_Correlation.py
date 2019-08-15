import numpy as np
import xarray as xr
import math
import matplotlib
import scipy.stats
import matplotlib.pyplot as plt

"""Scatterplot """

reg = 'center'
nemovar = 'ssh_'
distvar = 'kulk_'


dif = np.load('/scratch/judithe/tests/dif_kukulka_eularian_grid_025.npy')
nemo = np.load('/scratch/judithe/tests/ssh_eularian_grid_025.npy')
print(np.shape(dif))
print(np.shape(nemo))

if reg == 'natl':
    # North Atlantic
    dif = dif.flatten()
    nemo = nemo.flatten()
    
elif reg == 'gulf':
# Gulf stream (-80) - (-70) lons and 30 - 40 lats
    dif = (dif[140:180, 68:108]).flatten()
    nemo = (nemo[140:180, 68:108]).flatten()
#print(np.shape(dif))
#print(np.shape(nemo))

elif reg == 'labrador':
# Labrador sea (-60) - (-50) lons and 55 - 65 lats
    dif = (dif[40:80, 145:185]).flatten() 
    nemo = (nemo[40:80, 145:185]).flatten()
#print(np.shape(dif))
#print(np.shape(nemo))

elif reg == 'brazil':
# Brazil (-50) - (-70)lons and 0 - 10 lats
    dif = (dif[260:300, 188:268]).flatten() 
    nemo = (nemo[260:300, 188:268]).flatten() 
#print(np.shape(dif))
#print(np.shape(nemo))

elif reg == 'center': 
# center (-30) - (-60) lons and 30 - 60 lats
    dif = (dif[60:180, 148:268]).flatten() 
    nemo = (nemo[60:180, 148:268]).flatten() 
#print(np.shape(dif))
#print(np.shape(nemo))


print(np.shape(dif))
print(np.shape(nemo))


dif9 = np.argwhere(dif <= -9999.0)
diff = np.delete(dif,dif9)

difnan = np.argwhere(np.isnan(diff))
difff = np.delete(diff,difnan)

print(np.shape(dif))
print(np.isnan(dif))

nemoo = np.delete(nemo,dif9)
nemooo = np.delete(nemoo,difnan)
nemonan = np.argwhere(np.isnan(nemooo))

NEMO = np.delete(nemooo,nemonan)
DIF = np.delete(difff,nemonan)


print(np.shape(NEMO))
print(NEMO.max())
print(np.shape(DIF))

print(np.argwhere(np.isnan(NEMO)))
print(np.argwhere(np.isnan(DIF)))
#Nemo[~np.isnan(Nemo)]


#   xnan = np.argwhere(np.isnan(t))
 #   t = np.delete(t,xnan)
 #   y = np.delete(y,xnan)
 #   y20 = np.argwhere(y<-19.9) #delete lower boundary values
 #   y0 = np.argwhere(y>-0.1)
 #   t = np.delete(t,y20)
 #   y = np.delete(y,y20)

corr, pv = scipy.stats.pearsonr(DIF, NEMO)

'''Histograms 2D: Velocity vs Depth'''

import netCDF4
from matplotlib.colors import LogNorm

font = {'family' : 'monospace',
        'weight' : 'normal',
        'size'   : 32}

fig, ax = plt.subplots(figsize = (14, 16))
matplotlib.rc('font', **font)
#ax.figure(figsize = (14, 14))
cb = ax.hist2d(DIF, NEMO, bins=100, norm=LogNorm(),cmap='plasma')
plt.xlabel('Difference surface and subsurface trajectories')
plt.xticks(rotation=45)
plt.ylabel('SSH [m]')
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
textstr = '$\it{r}$' + ' ' + '=' + ' ' + str(round(corr, 3))

# place a text box in upper left in axes coords
ax.text(0.05, 0.95 , textstr, fontsize=32, transform=ax.transAxes,
        verticalalignment='baseline', horizontalalignment = 'left', bbox=props)

#ax.text(right, top, 'right top',
#        horizontalalignment='right',
#        verticalalignment='top',
#        transform=ax.transAxes)
cax = plt.axes([0.17, 0.94, 0.7, 0.02])

cbar = plt.colorbar(cb[3], cax=cax, orientation = 'horizontal')
cbar.ax.tick_params(labelsize=32, rotation = 45)
#clabtix = cbar.ax.get_xticklabels()
#cbar.ax.set_xticklabels(clabtix, rotation=10)

fig1 = plt.gcf()
plt.show()
plt.tight_layout()

#fig1.savefig('/scratch/judithe/Graphs/2dhist_corr_' + nemovar + distvar + reg + '.png')
