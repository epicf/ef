# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:24:39 2017

@author: ab
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from scipy.interpolate import griddata

def get_aprox(h5_file_in):
    h5file = h5py.File(h5_file_in,'r')
    data = h5file["data"]
    return data[:,0:6]

x=get_aprox('B.h5')

grid_x, grid_y, grid_z = np.mgrid[0:2:21j,0:2:21j,0:40:101j]

points = x[:,0:3]
values = [x[:,3],x[:,4],x[:,5]]
B_abs = np.sqrt(values[0]*values[0]+values[1]*values[1]+values[2]*values[2])

grid_xx = griddata(points, values[0], (grid_x, grid_y, grid_z), method='nearest')
grid_yy = griddata(points, values[1], (grid_x, grid_y, grid_z), method='nearest')
grid_zz = griddata(points, values[2], (grid_x, grid_y, grid_z), method='nearest')

grid_abs = griddata(points, B_abs , (grid_x, grid_y, grid_z), method='nearest')

plt.show()
plt.figure(figsize=(10,10))
#plt.imshow(grid_abs[:,20,:])
plt.streamplot(grid_z[:,10,:],grid_x[:,10,:],grid_zz[:,10,:],grid_xx[:,10,:], density=4.5)



os.chdir("./")
filename = 'out_test_0*00.h5'

x_num=[]
y_num=[]
z_num=[]

for f in glob.glob(filename):
    h5file = h5py.File( f, mode = "r" )
        
        
    x_num=np.insert(x_num, 0, h5file["/Particle_sources/cathode_emitter/position_x"])
    y_num=np.insert(y_num, 0, h5file["/Particle_sources/cathode_emitter/position_y"])
    z_num=np.insert(z_num, 0, h5file["/Particle_sources/cathode_emitter/position_z"])
    

    h5file.close()

#plt.show()
#plt.xlabel( "Z [cm]" )
#plt.ylabel( "Y [cm]" )
#plt.xlim(np.min(z_num),np.max(z_num))
plt.xlim(20,40)
plt.ylim(0,2)
plt.plot( z_num, y_num, '.', label = "num" )
#plt.legend()
plt.savefig( 'beam_contourZY.png' )
plt.close()

'''
    plt.xlabel( "Z [cm]" )
    plt.ylabel( "X [cm]" )
    plt.plot( z_num, x_num, '.', label = "num" )
    plt.legend()
    plt.savefig( f+"beam_contourZX.png" )
    plt.close()
    plt.xlabel( "Z [cm]" )
    plt.ylabel( "X [cm]" )
    plt.plot( z_num, y_num, '.', label = "num" )
    plt.legend()
    plt.savefig( f+"beam_contourZY.png" )
    plt.close()
'''
