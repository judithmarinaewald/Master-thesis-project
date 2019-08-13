"""
Judith Ewald
######################################################################
Code for the computation of particle trajectories
Advection of particles with nemo 1/12 degree. 
Surface Simulation
Particles initially as regular grid in North Atlantic
"""

#%matplotlib inline
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ScipyParticle, ErrorCode
import numpy as np
import math
from datetime import datetime, timedelta
from operator import attrgetter
import xarray as xr
from glob import glob
from argparse import ArgumentParser
from os import path
import warnings
warnings.filterwarnings("ignore")

"""Variables"""

# insert requestes lat/lon values for fieldset here:
outname = 'Surface'

minlat = 0
maxlat = 75
minlon = -100
maxlon = 30

init_grid_res = 0.2
griddir = '/scratch/judithe/Natl02grid/' # '/scratch/judithe/Natl1grid/', '/scratch/judithe/Natl02grid/'
pos = all
simdays = 7

outputdir = '/scratch/judithe/'
outfile = outputdir + outname + '_res' + str(init_grid_res) + '_simdays' + str(simdays) + '_pos' + str(pos) + '_fin' + '_dt3h_no_w'


"""Kernels and functions"""

def DeleteParticle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    particle.delete()
    
def kernel_track_velocity_2D(particle, fieldset, time):
    """Kernel to track particles in m/s"""
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    particle.u = u1 * 1852. * 60. * math.cos(particle.lat * math.pi/180.) # degrees / sec to meter / sec 
    particle.v = v1 * 1852. * 60.
    
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

#print(lat)
#print(lon)
#print(lat[:,1])
#print(lon[1,:])

# extract lat/lon values (in degrees) to numpy arrays
latvals = Lat[:]; lonvals = Lon[:] 

iy_min, ix_min = getclosest_ij(latvals, lonvals, minlat, minlon)
iy_max, ix_max = getclosest_ij(latvals, lonvals, maxlat, maxlon)

print('Index for latitude %s\xb0 = %s' % (minlat, iy_min))
print('Index for latitude %s\xb0 = %s' % (maxlat, iy_max))
print('Index for latitude %s\xb0 = %s' % (minlon, ix_min))
print('Index for latitude %s\xb0 = %s' % (maxlon, ix_max))

#lons, lats = np.meshgrid(range(ix_min, iy_max),range(iy_min, iy_max)) #regular meshgrid to release particles from


"""Set up fieldset"""

data_path = '/data2/imau/oceanparcels/hydrodynamic_data/NEMO-MEDUSA/ORCA0083-N006/'
ufiles = sorted(glob(data_path+'means/ORCA0083-N06_2010????d05U.nc'))
vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
meshfile = data_path+'domain/coordinates.nc'

filenames = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
             'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}

variables = {'U': 'uo', 'V': 'vo'}

dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
              'V': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}}

indices = {'lon': range(ix_min, ix_max), 'lat': range(iy_min, iy_max)} 

fieldset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True, indices = indices)

fieldset.U.vmax = 10 # to fix NEMO land bug: sometimes NAN value but sometiimes very high value!
fieldset.V.vmax = 10

print('Fieldset complete')

"""Particleset and execution"""

#Define a new Particle type including extra Variables
class VeloParticle(JITParticle):
    u = Variable('u', dtype=np.float32)
    v = Variable('v', dtype=np.float32)

#create particle set
#subsets of North Atlantic
#lons = np.load(griddir + 'Lons' + str(pos) + '.npy')
#lats = np.load(griddir + 'Lats' + str(pos) + '.npy') 


#whole North Atlantic
lons = np.load(griddir + 'Lons_full02' + '.npy')
lats = np.load(griddir + 'Lats_full02' + '.npy') 
times = [datetime(year=2010,month=1,day=10)]*len(lons) 

depths = [0.]*len(lons)
#print ('Number of particles: %s' % len(lons))

pset = ParticleSet(fieldset=fieldset, pclass=VeloParticle, lon=lons, lat=lats, time=times) #pclass = VeloParticle OR JITParticle

#pset.show(field=fieldset.U)

kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(kernel_track_velocity_2D)

"""Now execute the kernels for 30 days, saving data every 30 minutes"""
pset.execute(kernels, runtime=timedelta(days=simdays), dt=timedelta(minutes=10), 
             output_file=pset.ParticleFile(name=outfile, outputdt=timedelta(hours=3)), 
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

