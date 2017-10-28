import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


### Plot as histogram

filename = 'thermolization_0000100.h5'
h5file = h5py.File( filename, mode = "r" )

p_x_num = h5file["/Particle_sources/ions/momentum_x"]
p_y_num = h5file["/Particle_sources/ions/momentum_y"]
p_z_num = h5file["/Particle_sources/ions/momentum_z"]
v_prll = p_z_num / M_i
v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / M_i

plt.figure()
plt.xlabel( "V_prll [cm/s]" )
plt.ylabel( "N of Ions" )
plt.hist( v_prll )
#plt.ylim( 0, x_volume_size )
#plt.xlim( 0, z_volume_size )
#plt.plot( z_num, r_num, '.', label = "num" )
#plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
#plt.plot( z_an, r_an_lower, color = "g" )
#plt.legend()
plt.savefig( "ion_velocities_hist.png" )
h5file.close()
