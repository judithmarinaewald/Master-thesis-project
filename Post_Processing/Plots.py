'''Plotting functions'''

import netCDF4
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
from matplotlib.colors import LogNorm


'''North Atlantic'''

'''Prepare Plotting'''
font = {'family' : 'italic',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

def North_Atlantic():
    mp = plt.axes(projection=ccrs.PlateCarree())
    mp.set_xticks([-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30], crs=ccrs.PlateCarree())
    mp.set_yticks([0, 10, 20, 30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    mp.xaxis.set_major_formatter(lon_formatter)
    mp.yaxis.set_major_formatter(lat_formatter)
    mp.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
    mp.set_extent([-100, 50, 0, 75], ccrs.PlateCarree()) #this sets the dimensions of your map (west-east-norht-south extent)
    mp.coastlines()
    mp.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='black'))
    
North_Atlantic()


'''North Atlantic Gulf stream'''

def North_Atlantic_Gulf():
    mp = plt.axes(projection=ccrs.PlateCarree())
    mp.set_xticks([-85, -80, -75, -70, -65, -60], crs=ccrs.PlateCarree())
    mp.set_yticks([25, 30, 35, 40, 45], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    mp.xaxis.set_major_formatter(lon_formatter)
    mp.yaxis.set_major_formatter(lat_formatter)
    mp.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
    mp.set_extent([-85, -60, 25, 45], ccrs.PlateCarree()) #this sets the dimensions of your map (west-east-norht-south extent)
    mp.coastlines()
    mp.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='black'))

North_Atlantic_Gulf()


'''Maps: Particle Velocities/ Depths'''

def Map_Pset_Vel_Depths(name, dset, var, obs, cblim_min, cblim_max, tix, cblabel, outdir):

    nc = netCDF4.Dataset(dset)
    x = nc.variables["lon"][:]
    y = nc.variables["lat"][:]
    z = nc.variables[var][:]

    for i in range(len(x[0,0:obs])):
        fig = plt.figure(figsize=(13,10))
        North_Atlantic()
        #plt.title('North Atlantic' + '\n' + 'Particle Trajectories' + '\n' + name ) 
    
        if var == 'z':
            plt.scatter(x[:,0+i], y[:,0+i], c=-z[:,0+i], s=2)
            plt.set_cmap('viridis')
        else:
            plt.scatter(x[:,0+i], y[:,0+i], c=z[:,0+i], s=2)
            plt.set_cmap('plasma')
        
        cbar = plt.colorbar(extend = 'min') #,ticks=tix
        plt.clim(cblim_min, cblim_max)
        #cbar.ax.set_yticklabels(tix) #[0,-250,-500,-750]
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=18) 
        cbar.ax.set_xlabel(cblabel, fontsize = 18)
        plt.text(-97, 3, 'After' + ' ' + str(i*3) + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.5), fontsize = 18)
        plt.savefig(outdir + name + str(var) + str(i) + '.png')
        #print(str(i))
        plt.close(fig) 

Map_Pset_Vel_Depths(name = 'Kukulka_depth', dset = '/scratch/judithe/Kukulka_res0.2_simdays7_posall_dt3halldepths_winter.nc', var = 'z', obs = 57, 
                 cblim_min = -50, cblim_max = 0, tix = [-10,-20,-30,-40,-50], cblabel = 'Depth' + '\n' + '[m]', outdir = '/scratch/judithe/Graphs/Maps_depths/Kukulka/')

#tix = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]


'''Maps: Plot Polar Coordinates'''

def Map_Pset_Polar_Coord(name, dset_xy, dset_pc, var, clabel, outdir):

    nc = netCDF4.Dataset(dset_xy)
    lons = nc.variables["lon"][:]
    lats = nc.variables["lat"][:]

    for i in range(1,57):
        r = np.load(dset_pc + str(i) + '.npy')
        fig = plt.figure(figsize=(13,10))
        North_Atlantic_Gulf()
        plt.scatter(lons[:,i], lats[:,i], c=r, s=1)
        if var=='t':
            plt.set_cmap('hsv')
            plt.clim(0,360)
        else:
            plt.set_cmap('rainbow')
            plt.clim(0,1)
        cbar = plt.colorbar()
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_xlabel(clabel)
        #plt.title(name)
        plt.text(-97, 3, 'After' + ' ' + str(i*3) + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.5), fontsize = 14)
        plt.savefig(outdir + name + str(var) + str(i) + '.png')
        plt.close(fig)
    
Map_Pset_Polar_Coord(name = 'Poulain_scldist_smallgrid', dset_xy = '/scratch/judithe/Poulain270p_small_fs2.nc', 
                     dset_pc = '/scratch/judithe/Post_Parcels/Polar_coordinates/Small Grid/Poulain/from_initialp_scl_dist_57obs_012010obs', var = 'scl_dist', 
                     clabel = '[]', outdir = '/scratch/judithe/Graphs/Polar_coordinates/Small Grid/Poulain/')
                     
                     
 
'''Histograms: Amount of Particles with Depths'''
import matplotlib
font = {'family' : 'italic',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
#matplotlib.rcParams.update({'font.size': 12})
#matplotlib.rcParams.update({'font.style' : italic})

def Hist_Particles_Depth(name, dset, obs, outdir):
    
    ds = netCDF4.Dataset(dset)
    z = ds.variables["z"][:]

    hours = range(obs)
    hours = [i*3 for i in hours]

    fig, axs = plt.subplots(2,7, figsize=(15, 5), facecolor='w', edgecolor='k', sharex=True, sharey=True)
    axs = axs.ravel()
    
    for i, j in zip(range(0,56,4), range(0,15)):
        x = -z[:,i]
        xnan = np.argwhere(np.isnan(x))
        x = np.delete(x,xnan)
    
        cb = axs[j].hist(x,bins=100, log=True) #
        axs[j].set_title('After' +' '+ str(hours[i]) +' '+ 'hours', fontsize=12, weight='bold')
        if j>=7:
            axs[j].set_xlabel('$\it{Depth [m]}$', fontsize=14)
        if j==0 or j==7:
            axs[j].set_ylabel('$\it{N° Particles}$', fontsize=14)
        
    fig1 = plt.gcf()
    plt.show()
    plt.tight_layout()
    fig1.savefig(outdir + 'Histo_NParticles_Depths' + name + '_'  + '.png')

Hist_Particles_Depth(name = 'upvw_winter', dset='/scratch/judithe/uvwp_corr.nc', obs=56,
                    outdir = '/scratch/judithe/Graphs/Histograms/')
                    
                    
'''Histograms 2D: Velocity vs Depth'''

def Hist2D_Vel_vs_Depth(name, dset, obs, xvar, xlab, outdir):
    
    nc = netCDF4.Dataset(dset)
    xv = nc.variables[xvar][:]
    yv = nc.variables['z'][:]
    
    hours = range(obs)
    hours = [i*3 for i in hours]

    fig, axs = plt.subplots(2,7, figsize=(15, 6), facecolor='w', edgecolor='k', sharex=True, sharey=True)
    axs = axs.ravel()
    for i, j in zip(range(0,56,4), range(0,15)):
        
        x = xv.data[:,i]
        y = -yv.data[:,i]
        xnan = np.argwhere(np.isnan(x))
        x = np.delete(x,xnan)
        y = np.delete(y,xnan)
        
        cb = axs[j].hist2d(x,y,bins=40, norm=LogNorm(),cmap='plasma')
        axs[j].set_title('After' +' '+ str(hours[i]) +' '+ 'hours', fontsize=12, weight='bold')
        if j>=7:
            axs[j].set_xlabel(xlab, fontsize=14)
        if j==0 or j==7:
            axs[j].set_ylabel('Depth (m)', fontsize=14)
        
    plt.subplots_adjust(bottom=0.1, right=0.83, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.02, 0.8])
    plt.colorbar(cb[3], cax=cax)
    fig1 = plt.gcf()
    plt.show()
    plt.tight_layout()
    fig1.savefig(outdir + name + '_' + str(xvar) + '.png')

Hist2D_Vel_vs_Depth(name = 'Poulain_winter', obs = 57, dset = '/scratch/judithe/Poulain_res0.2_simdays7_posall_dt3h_no_w_alldepths_winter.nc', 
                      xvar = 'v', xlab = 'v (m/s)', outdir = '/scratch/judithe/Graphs/Histograms/Hist2d/Poulain/winter/' )
                      
                      
'''Histograms 2D: Polar Coordinates vs Depth'''

font = {'family' : 'italic',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

def Hist2d_PolarC_vs_Depth (name, obs, dset_pc, dset_x, xlab, outdir):
    
    dset = xr.open_dataset(dset_x, decode_times=False)
    hours = range(obs)
    hours = [i*3 for i in hours]

    fig, axs = plt.subplots(2,7, figsize=(15, 6), facecolor='w', edgecolor='k', sharex=True, sharey=True)
    axs = axs.ravel()
    for i, j in zip(range(0,56,4), range(0,15)):
        var = np.load(dset_pc + str(i+1) + '.npy')
        z = dset.z[:,i]
        y = -z.data
        xnan = np.argwhere(np.isnan(var))
        var = np.delete(var,xnan)
        y = np.delete(y,xnan)
    
        cb = axs[j].hist2d(var,y,bins=40, norm=LogNorm(),cmap='summer')
        axs[j].set_title('After' +' '+ str(hours[i]) +' '+ 'hours', fontsize=12, weight ='bold')
        if j>=7:
            axs[j].set_xlabel(xlab, fontsize=14)
        if j==0 or j==7:
            axs[j].set_ylabel('Depth [m]', fontsize=14)
        
    plt.subplots_adjust(bottom=0.1, right=0.83, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(cb[3], cax=cax)
    cbar.set_label('$\it{N° Particles}$', rotation=90)
    fig1 = plt.gcf()
    plt.show()
    plt.tight_layout()
    fig1.savefig(outdir + name + '_' + str(xlab) + '.png')

Hist2d_PolarC_vs_Depth(name = 'Poulain_angle', obs = 57, dset_pc = '/scratch/judithe/Post_Parcels/Polar_coordinates/Poulain/winter/angle_012010obs', 
                       dset_x = '/scratch/judithe/Poulain_res0.2_simdays7_posall_dt3h_no_w_alldepths_winter.nc', xlab = '[°]', 
                       outdir = '/scratch/judithe/Graphs/Histograms/Hist2d/' )
                       

'''Maps: Plot Mean/Std Values'''

im = np.load('/scratch/judithe/tests/tau_eularian_grid_025.npy') #mean_dif_normdist_surf_poulain

im = np.ma.array(-im)
#mask values below a certain threshold
im_masked = np.ma.masked_where(im < -9000 , im)
plt.figure(figsize=(30,15))
North_Atlantic()
plt.imshow(im_masked, cmap ='ocean',  extent=[-97,30,0,75]) #,
#plt.clim(-500,0)
plt.colorbar #(extend = 'both')
#plt.clim(-500,0)
#plt.savefig('/scratch/judithe/tests/mdl_mean_025.png')


'''Maps: Plot Mean/Std Values'''

im = np.load('/scratch/judithe/tests/tau_eularian_grid_025.npy') #mean_dif_normdist_surf_poulain

#im = im[40:80, 145:185] # lbrador
#im = im[140:180, 68:108] #Gulf 
#im = im[260:300, 188:268] #Brazil 
im = im[60:180, 148:268] #center

im = np.ma.array(-im)
#mask values below a certain threshold
im_masked = np.ma.masked_where(im < -9000 , im)
plt.figure(figsize=(30,15))
North_Atlantic()

#plt.imshow(im_masked, cmap ='ocean',  extent=[-60,-50,55,65]) #labrador
#plt.imshow(im_masked, cmap ='ocean',  extent=[-80,-70,30,40]) #Gulf
#plt.imshow(im_masked, cmap ='ocean',  extent=[-50,-30,0,10]) #Brazil
plt.imshow(im_masked, cmap ='ocean',  extent=[-60,-30,30,60]) #center

plt.colorbar #(extend = 'both')
#plt.clim(-0.1,0.1)
#plt.savefig('meandif_surf_poulain.png')


'''Maps: Particle Trajectories'''

def Map_Pset_Vel_Depths_Gulf(name, dset, var, obs, cblim_min, cblim_max, tix, cblabel, outdir):

    nc = netCDF4.Dataset(dset)
    x = nc.variables["lon"][:]
    y = nc.variables["lat"][:]
    z = nc.variables[var][:]

    #for i in range(len(x[0,0:obs])):
    fig = plt.figure(figsize=(13,10))
    North_Atlantic_Gulf()
    #plt.title('North Atlantic' + '\n' + 'Particle Trajectories' + '\n' + name ) 
    
    if var == 'z':
        plt.scatter(x[:], y[:], c=-z[:], s=10)
        plt.set_cmap('viridis')
    else:
        plt.scatter(x[:], y[:], c=z[:], s=10)
        plt.set_cmap('plasma')
        
    cbar = plt.colorbar(extendfrac='uniform',ticks=tix) 
    #plt.clim(cblim_min, cblim_max)
    cbar.ax.set_yticklabels(tix) #[0,-250,-500,-750]
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel(cblabel)
    #plt.text(-97, 3, 'After' + ' ' + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.5), fontsize = 14)
    plt.savefig(outdir + name + str(var) + '.png')
    #print(str(i))
    plt.close(fig) 

Map_Pset_Vel_Depths_Gulf(name = '270p_poulain', dset = '/scratch/judithe/Poulain270p_small_fs2.nc', var = 'z', obs = 57, 
                 cblim_min = 0, cblim_max = -750, tix = [0,-250,-500,-750], cblabel = '[m]', outdir = '/scratch/judithe/tests/')

#tix = [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2], [0,-250,-500,-750]



