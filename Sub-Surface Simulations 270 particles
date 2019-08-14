"""
Judith Ewald 
######################################################################
Code for the computation of particle trajectories
Advection of particles with nemo 1/12 degree. 
Simulation with wind-stress based mixing of Poulain et al. 2019
270 particles
"""

import xarray as xr
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle, Variable, AdvectionRK4, ErrorCode #_3D
from datetime import timedelta
from datetime import datetime
from glob import glob
import math
#import warnings
#warnings.filterwarnings("ignore")
#import netCDF4

"""Variables"""

# insert requestes lat/lon values for fieldset here:
outname = 'Poulain'

minlat = 25
maxlat = 45
minlon = -85
maxlon = -60

init_grid_res = 0.2
griddir = '/scratch/judithe/judithe/Natl02grid/' # '/scratch/judithe/Natl1grid/', '/scratch/judithe/Natl02grid/'
pos = 'all'
simdays = 7

outputdir = '/scratch/judithe/'
outfile = outputdir + outname + '270p' + '_small_fs' +'2'


#Define rising velocity for vertical mixing (Poulain et al. 2019)
#pp = 1.005 #particle density; ellipsoidal shape: 1.005; spherical shape: 0.9
#V = 0.0025 #particle volume; LMP = 0.1 - 0.5, SMP = 0.0025 - 0.1
wrise = 0.00049 # ellipsoidal and smallest SMP value


"""Kernels and functions"""

def kernel_poulain_mixing(particle, fieldset, time):  
    """
    :Kernel that randomly distributes particles along the vertical according to an expovariate distribution.
    :Parameterization according to Poulain et al. (2019): The effect of wind mixing on the vertical distribution of buoyant plastic debris
    :Comment on dimensions: tau needs to be in Pa
    """
    stress = fieldset.TAU[time,0.,particle.lat,particle.lon] #TAU is stress in Pa taken from T from NEMO dataset
    A0=0.31 * math.pow(stress,1.5)                                      #turbulent eddy exchange coeff near surface
    l=fieldset.wrise/A0                                                 #calucate lambda
    d=random.expovariate(l)                                             #poulain formula. Used depths of [0 ... 10.] m
#    if d>200.:
#        particle.depth=200.
#    else:
    particle.depth=d

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()
    
def TrackVelocity3D(particle, fieldset, time): 
    """Kernel for tracking particles in m/s instead of degrees/sec."""
#    print("TIME : %g" % time)
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon] #
    particle.u = u1 * 1852. * 60. * math.cos(particle.lat * math.pi/180.) 
    particle.v = v1 * 1852. * 60.
#    particle.w = w1
    
def getclosest_ij(lats,lons,latpt,lonpt):     
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats-latpt)**2 + (lons-lonpt)**2                 # find squared distance of every point on grid
    minindex_flattened = dist_sq.argmin()                       # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened, lats.shape)     # Get 2D index for latvals and lonvals arrays from 1D index

print('Kernels complete')

"""Find lon/lat indices for fieldset"""

initialgrid_mask = '/data2/imau/oceanparcels/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/means/ORCA0083-N06_20101206d05U.nc'
mask = xr.open_dataset(initialgrid_mask, decode_times=False)
Lat, Lon = mask.variables['nav_lat'], mask.variables['nav_lon']

# extract lat/lon values (in degrees) to numpy arrays
latvals = Lat[:]; lonvals = Lon[:] 

iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)

print('Index for latitude %s\xb0 = %s' % (minlat, iy_min))
print('Index for latitude %s\xb0 = %s' % (maxlat, iy_max))
print('Index for longitude %s\xb0 = %s' % (minlon, ix_min))
print('Index for longitude %s\xb0 = %s' % (maxlon, ix_max))

#lons, lats = np.meshgrid(range(ix_min, iy_max),range(iy_min, iy_max)) #regular meshgrid to release particles from

#times = [datetime(year=2010,month=1,day=1)]*len(lons) 
#depths = [0.]*len(lons)

#print(lons)
#print(lats)
#print(times)
#print(depths)
#print('Meshgrid complete')

"""Set up fieldset"""

data_path = '/data2/imau/oceanparcels/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/'
ufiles = sorted(glob(data_path+'means/ORCA0083-N06_201001*d05U.nc'))
vfiles = sorted(glob(data_path+'means/ORCA0083-N06_201001*d05V.nc'))
wfiles = sorted(glob(data_path+'means/ORCA0083-N06_201001*d05W.nc'))
taufiles = sorted(glob(data_path+'means/ORCA0083-N06_201001*d05T.nc'))


mesh_mask = data_path + 'domain/coordinates.nc'

filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
             'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
             #'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles},
             'TAU': {'lon': mesh_mask, 'lat': mesh_mask, 'data': taufiles}} 
                
variables = {'U': 'uo',
             'V': 'vo',
             #'W': 'wo',
             'TAU': 'taum'}

dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              #'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
              'TAU': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}

indices = {'lon': range(ix_min, ix_max), 'lat': range(iy_min, iy_max)}  #, 'depth': range(0, 50)

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, indices=indices)

#def compute(fieldset):
#    fieldset.W.data[:, 0, :, :] = 0.
#    fieldset.compute_on_defer = compute
#    return fieldset

fieldset.wrise = wrise

fieldset.U.vmax = 10 # to fix NEMO land bug: sometimes NAN value but sometiimes very high value!
fieldset.V.vmax = 10
fieldset.TAU.vmax = 10


#for fld in [fieldset.U, fieldset.V, fieldset.W, fieldset.TAU]:
#    print('%s: %s' % (fld.name, fld.units))


print('Fieldset complete')


"""Particleset and execution"""

class VeloParticle3D(JITParticle):
    """Particle class to track velocities in m/s"""
    u = Variable('u', dtype=np.float32)
    v = Variable('v', dtype=np.float32)
#    w = Variable('w', dtype=np.float32)
    
#lons = np.load(griddir + 'Lons_full02' + '.npy')
#lats = np.load(griddir + 'Lats_full02' + '.npy') 

lons1 = np.arange(-80, -64, 0.1) 
lons2 = np.arange(-75, -64, 0.1)
lons = np.concatenate((lons1, lons2), axis=None)

lats1 = np.full((len(lons1)), 30)
lats2 = np.full((len(lons2)), 35)
lats = np.concatenate((lats1, lats2), axis=None)

#lons = np.load(griddir + 'Lons' + str(pos) + '.npy')
#lats = np.load(griddir + 'Lats' + str(pos) + '.npy')
#lons = [-90]
#lats = [25]
#for i in range(len(lons)):
#    print (lons[i], lats[i])
#exit(0)

times = [datetime(year=2010,month=1,day=10)]*len(lons) 
#print(times)

depths = [0.]*len(lons)
#print ('Number of particles: %s' % len(lons))

pset = ParticleSet(fieldset=fieldset, pclass=VeloParticle3D, lon=lons, lat=lats, depth=depths, time=times) # time default is 0
#pset.show(field=fieldset.U)

kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(TrackVelocity3D) + pset.Kernel(kernel_poulain_mixing) #

#Trajectory computation
#print (pset[0])
pset.execute(kernels, runtime=timedelta(days=simdays), dt=timedelta(minutes=10), 
             output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(hours=3)), 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

print('complete')
#print pset[0]
