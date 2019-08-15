import xarray as xr
import numpy as np
import math
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LogNorm
from haversine import haversine
import matplotlib

%matplotlib inline

'''Prepare Plotting'''
font = {'family' : 'italic',
        'weight' : 'normal',
        'size'   : 26}

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
    mp.set_extent([-100, 30, 0, 75], ccrs.PlateCarree()) #this sets the dimensions of your map (west-east-norht-south extent)
    mp.coastlines()
    mp.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='black'))
    
North_Atlantic()


'''North Atlantic Gulf stream'''

'''Prepare Plotting'''
font = {'family' : 'italic',
        'weight' : 'normal',
        'size'   : 22}

def North_Atlantic_Gulf():
    mp = plt.axes(projection=ccrs.PlateCarree())
    mp.set_xticks([-85, -80, -75, -70, -65, -60], crs=ccrs.PlateCarree())
    mp.set_yticks([25, 30, 35, 40], crs=ccrs.PlateCarree())
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    mp.xaxis.set_major_formatter(lon_formatter)
    mp.yaxis.set_major_formatter(lat_formatter)
    mp.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
    mp.set_extent([-85, -60, 25, 40], ccrs.PlateCarree()) #this sets the dimensions of your map (west-east-norht-south extent)
    mp.coastlines()
    mp.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='black'))
    
North_Atlantic_Gulf()

"""Plot Polar Coordinates"""

%matplotlib inline
import netCDF4
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

nc = netCDF4.Dataset("/scratch/judithe/Surface_res0.2_simdays7_posall_fin_dt3h_no_w_winter.nc")

lons = nc.variables["lon"][:]#.squeeze()
lats = nc.variables["lat"][:]#.squeeze()


'''Plot Polar Coordiantes map - distance (km)'''
i = 24 # 24*timesteps (each timestep is 3 hours) results in 3 days

#for i == 24: #in range(1,57):
if i==24: 
    r = np.load('/scratch/judithe/Post_Parcels/Polar_coordinates/Differences/Dif_Surf_Kukulka_normdist' + str(i) + '.npy')
    #print(i)
    #name = str('Polar coordinates - Distance (km) - Surface')
    fig = plt.figure(figsize=(13,10))
    North_Atlantic()
    plt.scatter(lons[:,i], lats[:,i], c=r, s=1) # c=z[0,:],
    plt.set_cmap('RdBu')
    cbar = plt.colorbar(extend = 'both')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=24) 
    cbar.ax.set_xlabel('[]', fontsize =24)
    plt.clim(-0.1,0.1)
    #plt.title('North Atlantic - Ocotber 2010' +'\n'+ name + '\n' )
    plt.text(-98, 3, 'After' + ' ' + str(i*3) + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.9), fontsize = 22)
    plt.savefig('/scratch/judithe/Graphs/' + 'difsurfkulk' + 'normdist'+ str(i) + '.png')
    plt.close(fig)
    
'''Plot Polar Coordiantes map - t (angle)'''

nc = netCDF4.Dataset("/scratch/judithe/Surface_res0.2_simdays7_posall_fin_dt3h_no_w_winter.nc")

lons = nc.variables["lon"][:]#.squeeze()
lats = nc.variables["lat"][:]#.squeeze()


#for i in range(1,57):
if i==24: 
    t = np.load('/scratch/judithe/Post_Parcels/Polar_coordinates/Differences/Dif_Surf_Poulain_angle' + str(i) + '.npy')
    #print(i)
    #name = str('Polar coordinates - angle (theta) - Surface')
    fig = plt.figure(figsize=(13,10))
    North_Atlantic()
    plt.scatter(lons[:,i], lats[:,i], c=t, s=1)
    plt.set_cmap('jet')
    cbar = plt.colorbar()
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=20) 
    cbar.ax.set_xlabel('[\xb0]', fontsize = 24)
    plt.clim(0,180)
    #plt.title('North Atlantic - October 2010' +'\n'+ name + '\n' )
    plt.text(-98, 3, 'After' + ' ' + str(i*3) + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.9), fontsize = 22)
    plt.savefig('/scratch/judithe/Graphs/' + 'difsurfpoul' + 'angle'+ str(i) + '.png')
    plt.close(fig)
    
    '''Plot Polar Coordiantes map - difference'''
import cmocean

nc = netCDF4.Dataset("/scratch/judithe/Surface270p_small_fs.nc")

lons = nc.variables["lon"][:]#.squeeze()
lats = nc.variables["lat"][:]#.squeeze()

for i in range(1,57):
    r = np.load('/scratch/judithe/Post_Parcels/Polar_coordinates/Small Grid/Differences/Dif_Surf_Poulain_angle' + str(i) + '.npy') #+ str(i) 
#print(i)
#name = str('Surface - Kukulka')
    fig = plt.figure(figsize=(13,10))
    North_Atlantic_Gulf()
    plt.scatter(lons[:,i], lats[:,i], c=r, s=15) # c=z[0,:],
    plt.set_cmap('plasma')
    cbar = plt.colorbar() #extend = 'both',
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel('Difference' +'\n'+ '[°]', fontsize = 24)
    cbar.ax.tick_params(labelsize=20) 
    plt.clim(0,180)
#plt.title('North Atlantic' +'\n'+ name)
    plt.text(-84, 38, 'After' + str(i*3) + ' ' + 'h', bbox=dict(facecolor='white', alpha=0.5), fontsize = 22) #+ str(i*3)
    plt.savefig('/scratch/judithe/Graphs/Polar_coordinates/Small Grid/Difference/'  + 'angles' + str(i) +'.png') #str(i)
    plt.close(fig)
    
'''Calculate Polar Coordinates from initial point for each particle'''

import xarray as xr
import numpy as np
import math
import netCDF4
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LogNorm
from haversine import haversine
import sklearn
from sklearn import preprocessing

%matplotlib inline

'''Open and read trajectories file'''

dset = xr.open_dataset('/scratch/judithe/uvwp_corr02grid.nc', decode_times=False)
outdir = '/scratch/judithe/Post_Parcels/Polar_coordinates/uvwp/'
from math import *


R = 6371 #earth radius estimated in km
#min_max_scaler = preprocessing.MinMaxScaler()

for j in range(1,57):
    lon1 = dset.lon[:,0] #lons for all traj for initial grid (obs0)
    lon2 = dset.lon[:,j] #lons for all traj for obs1
    lat1 = dset.lat[:,0] #lats for all traj for initial grid (obs0)
    lat2 = dset.lat[:,j] #lats for all traj for obs1
    #z1 = dset.z[:,i]
    #z2 = dset.z[:,j]
    x = np.subtract(lon2, lon1)
    y = np.subtract(lat2, lat1)
    #print(x)
    t = np.degrees(np.arctan2(y,x)) #calculate angle
    #bearing = atan2(sin(lon2-lon1)*cos(lat2), cos(lat1)*sin(lat2)in(lat1)*cos(lat2)*cos(lon2-lon1))
    #bearing = degrees(bearing)
    #bearing = (bearing + 360) % 360

    #r = np.sqrt(x**2+y**2) #calculate distance in degrees 
      
    # haversine formula to calculate distance in km
    s_lat = lat1*np.pi/180.0                      
    s_lng = np.deg2rad(lon1)     
    e_lat = np.deg2rad(lat2)                       
    e_lng = np.deg2rad(lon2)  
    
    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
    
    km = 2 * R * np.arcsin(np.sqrt(d)) #this is the distance projected on the surface
    
    #rel_dist = km / km.mean() #compute relative distance (standardized)
    std_dist = (km - km.mean())/km.std() #compute relative distance (standardized)
    #scl_dist = sklearn.preprocessing.minmax_scale(km, feature_range=(0, 1), axis=0, copy=False)
    norm_dist = (km - km.min()) / (km.max() - km.min())  
    
    #print(np.nanmax(rel_dist))
    #print(np.nanmin(rel_dist))
    #print(np.nanmax(std_dist))
    #print(np.nanmin(std_dist))
    #print(km)
    #print(km)
    #print(scl_dist)
    #print(scl_dist) 
    
    np.save(outdir + 'angle'  + '_012010' + 'obs' + str(j), t)
    np.save(outdir + 'km' + '_012010' + 'obs' + str(j), km)
    #np.save(outdir + 'rel_dist' + '_012010' + 'obs' + str(j), rel_dist)
    np.save(outdir + 'std_dist' + '_012010' + 'obs' + str(j), std_dist)
    #np.save(outdir + 'scl_dist' + '_012010' + 'obs' + str(j), scl_dist)
    np.save(outdir + 'norm_dist' + '_012010' + 'obs' + str(j), norm_dist)
    
'''Substract Surface from Poulain/ Kukulka - ANGLE'''

def Dif_surface_subsurf(name, obs, ddir_surf, ddir_subsurf, outdir):
    for i in range(1,obs):
        km_surface = np.load(ddir_surf + str(i) + '.npy') 
        km_poulain = np.load(ddir_subsurf + str(i) + '.npy')

        dif_surface_poulain = abs(np.subtract(km_surface,km_poulain)) % 180
        #print(dif_surface_poulain)
        #print(np.nanmax(dif_surface_poulain))
        #print(np.nanmin(dif_surface_poulain))
        np.save(outdir + name + str(i) + '.npy', dif_surface_poulain)
    
Dif_surface_subsurf(name = 'Dif_uvwk_Kukulka_angle', obs=57, ddir_surf='/scratch/judithe/Post_Parcels/Polar_coordinates/Kukulka/winter/angle_012010obs',
                   ddir_subsurf = '/scratch/judithe/Post_Parcels/Polar_coordinates/uvwk/angle_012010obs',
                    outdir = '/scratch/judithe/Post_Parcels/Polar_coordinates/Differences/')

'''Substract Surface from Poulain/ Kukulka - DISTANCE'''

def Dif_surface_subsurf(name, obs, ddir_surf, ddir_subsurf, outdir):
    for i in range(1,obs):
        km_surface = np.load(ddir_surf + str(i) + '.npy') 
        km_poulain = np.load(ddir_subsurf + str(i) + '.npy')

        dif = (np.subtract(km_surface,km_poulain)) 
        dif[np.isnan(dif)] = 0
        dif2 = (dif - dif.mean()) / dif.std()
        #print(dif)
        #print(dif2)
        
        #print(dif_surface_poulain)
        #print(np.nanmax(dif_surface_poulain))
        #print(np.nanmin(dif_surface_poulain))
        np.save(outdir + name + str(i) + '.npy', dif)
        #del dif
        #del dif2
    
Dif_surface_subsurf(name = 'Dif_uvwp_Poulain_km', obs=57, ddir_surf='/scratch/judithe/Post_Parcels/Polar_coordinates/Poulain/winter/km_012010obs',
                   ddir_subsurf = '/scratch/judithe/Post_Parcels/Polar_coordinates/uvwp/km_012010obs',
                    outdir = '/scratch/judithe/Post_Parcels/Polar_coordinates/Differences/')
   
    
