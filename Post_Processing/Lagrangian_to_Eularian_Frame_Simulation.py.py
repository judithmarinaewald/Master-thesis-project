'''Calculate 1x1 deg grid for traj output file with means of u, v velocities for each grid'''

import numpy as np
import xarray as xr
import math

name = 'Surface_winter'
outdir = '/scratch/judithe/Post_Parcels/Means/Velocities/Surface/winter/'
variable = 'v'
ds = xr.open_dataset('/scratch/judithe/Surface_res0.2_simdays7_posall_fin_dt3h_no_w_winter.nc', decode_times=False)
ds.load()

print('data loaded in memory')

# https://github.com/pydata/xarray/issues/1732# print(ds)

def find_indices(ds, lat1, lat2, lon1, lon2, obs):  # find indices for final observation (last timesstep and all trajectories)
    '''
    ds = dataset
    lon1 and lon2 between -180, +180
    '''
    lon = ds.lon.isel(obs=obs)
    lat = ds.lat.isel(obs=obs)
    #
    # if len(lon) == 0 or len(lat) == 0:
    #     return np.empty((0))
    if lon2 < lon1:
        lon2 += 360
        lon = lon.where(lon >= 0, lon + 360)
    indices = np.where((lat > lat1) & (lat < lat2) & (lon > lon1) & (lon < lon2))[0]
    return indices


#print('start test search and mean')
#temp1 = ds['v'][find_indices(ds, 10, 11, -31, -30, obs=-1)]
#temp = temp1.mean()  # call to calculate mean of u velocity for indices
#print('test search completed')
#print('mean: ', temp)

'''loop over function find_indices to find incdices and calculate mean of u/ v velocities for a 1x1 deg grid of the North Atlantic
for all obs and save as .npy array'''


#columns will represent longitudes,rows- lat
#to help with the visualization as well.

for obs in range(0,56): 
    
    #North Atlantic
    start_lon=-97
    end_lon=30
    start_lat=0
    end_lat=75
    
    columns = abs(start_lon - end_lon)
    rows = abs(start_lat - end_lat)
    out = np.full((rows, columns), -9999.0) # -9999.0 to differentiate between real 0 and not filled sections

    print('start mean loop')
    # start with the upper left corner of the area
    current_lat = start_lat
    current_lon = start_lon

    for i in range(0, rows):  # rows= latitudes
        current_lon = start_lon  # reset the longitude to the left
        for j in range(0, columns):  # columns= longitudes

            # print('trying for position: ', current_lat, current_lat - 1, current_lon, current_lon + 1)
            # find_indices assumes that lat1<lat2,
            # hence passing the lower latitude(current_lat - 1) before current_lat
            indices = find_indices(ds, current_lat - 1, current_lat, current_lon, current_lon + 1, obs)

            # not all searches return indices
            if len(indices) != 0:
                temp = ds[variable][indices]
                out[i][j] = temp.mean()  # find mean u velocity for all traj of each gridcell
                #print('mean added')
                
            current_lon = current_lon + 1  # moving from  left boundary to right boundary

        current_lat = current_lat - 1  # moving from  upper boundary to lower boundary
        
        out = np.array(out)    
        np.save(outdir + 'mean' + '_' + name + '_' + variable + str(obs) + '.npy', out)    
    
print('process completed')
# you can put a debug point here and view your array or print- easier for a small matrix
#print('print test a value: ', out[1][4])
