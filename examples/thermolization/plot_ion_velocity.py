import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = 'thermolization_0000100.h5'
h5file = h5py.File( filename, mode = "r" )

def get_electrons_current( h5file ):
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    charge = h5file["/Particle_sources/electrons"].attrs["charge"][0]
    particles_per_step = h5file[
        "/Particle_sources/electrons"].attrs["particles_to_generate_each_step"][0]
    current = particles_per_step * charge / time_step
    return current
    
def get_ions_parameters( h5file ):
    mass = h5file["/Particle_sources/ions"].attrs["mass"][0]
    charge = h5file["/Particle_sources/ions"].attrs["charge"][0]
    return ( mass, charge )

#I = get_electrons_current( h5file )
m, q = get_ions_parameters( h5file )

p_x_num = h5file["/Particle_sources/ions/momentum_x"]
p_y_num = h5file["/Particle_sources/ions/momentum_y"]
p_z_num = h5file["/Particle_sources/ions/momentum_z"]
v_prll = p_z_num / m
v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / m

z_volume_size = h5file["/Spatial_mesh"].attrs["z_volume_size"][0]
x_volume_size = h5file["/Spatial_mesh"].attrs["x_volume_size"][0]

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



### As points 


def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("thermolization_[0-9]*.h5"):
        h5files.append( file )
    return h5files

out_files = find_necessary_out_files()
out_files.sort()
print( out_files )

plt.figure()
plt.xlabel( "V_prll [cm/s]" )
plt.ylabel( "N of saved file" )
plt.yticks( range( len(out_files) ) )
# todo: show current time on yticks
# time_step = h5file["/Time_grid"].attrs["current_time"][0]


for time_step_num, filename in enumerate(out_files):
    h5file = h5py.File( filename, mode = "r" )
    p_x_num = h5file["/Particle_sources/ions/momentum_x"]
    p_y_num = h5file["/Particle_sources/ions/momentum_y"]
    p_z_num = h5file["/Particle_sources/ions/momentum_z"]
    v_prll = p_z_num / m
    v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / m

    #plt.ylim( 0, x_volume_size )
    #plt.xlim( 0, z_volume_size )
    y = np.full_like( v_prll, time_step_num )
    plt.plot( v_prll, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )
    v_prll_mean = np.mean( v_prll )
    v_prll_stdiv = np.std( v_prll )
    plt.text( 3.1e5, time_step_num,
              "mean={:.2e}".format( v_prll_mean ),
              fontsize = 10 )
    plt.text( 3.1e5, time_step_num - 0.3,
              "stdiv={:.2e}".format( v_prll_stdiv ),
              fontsize = 10 )
    #plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
    #plt.plot( z_an, r_an_lower, color = "g" )
    h5file.close()

#plt.legend()
plt.savefig( "ion_velocities_as_points.png" )

