import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import operator

def get_electrons_parameters( h5file ):
    mass = h5file["/Particle_sources/electrons"].attrs["mass"][0]
    charge = h5file["/Particle_sources/electrons"].attrs["charge"][0]
    return ( mass, charge )

def get_electrons_beam_radius( h5file ):
    return h5file["/Particle_sources/electrons"].attrs["cylinder_radius"][0]

def get_electrons_current( h5file ):
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    charge = h5file["/Particle_sources/electrons"].attrs["charge"][0]
    particles_per_step = h5file[
        "/Particle_sources/electrons"].attrs["particles_to_generate_each_step"][0]
    current = np.abs( particles_per_step * charge / time_step )
    return current

def get_electrons_current_density( h5file ):
    I = get_electrons_current( h5file )
    radius = h5file["/Particle_sources/electrons"].attrs["cylinder_radius"][0]
    current_density = I / ( np.pi * radius * radius )
    return current_density

def get_electrons_concentration( h5file ):
    # particles_per_step = h5file[
    #     "/Particle_sources/electrons"].attrs["particles_to_generate_each_step"][0]
    #time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    total_particles = len( h5file["/Particle_sources/electrons/particle_id"] )
    radius = h5file["/Particle_sources/electrons"].attrs["cylinder_radius"][0]    
    z_start = h5file["/Particle_sources/electrons"].attrs["cylinder_axis_start_z"][0]
    z_end = h5file["/Particle_sources/electrons"].attrs["cylinder_axis_end_z"][0]
    source_volume = np.abs( z_end - z_start ) * np.pi * radius ** 2
    n_e = total_particles / source_volume
    return n_e

def get_electrons_energy( h5file ):
    pz = h5file["/Particle_sources/electrons"].attrs["mean_momentum_z"][0]
    m = h5file["/Particle_sources/electrons"].attrs["mass"][0]
    E = pz * pz / ( 2.0 * m )
    return E

def get_electrons_velocity( h5file ):    
    pz = h5file["/Particle_sources/electrons"].attrs["mean_momentum_z"][0]
    m = h5file["/Particle_sources/electrons"].attrs["mass"][0]
    v_e = pz / m
    return v_e

def get_ions_parameters( h5file ):
    mass = h5file["/Particle_sources/ions"].attrs["mass"][0]
    charge = h5file["/Particle_sources/ions"].attrs["charge"][0]
    return ( mass, charge )

def eval_a_prll( m_e, q_e, M_i, Q_i, v_e, j, R ):
    rho_0 = np.abs( q_e * Q_i / ( m_e * v_e ** 2 ) )
    tmp1 = 2.0 * np.pi * Q_i / M_i * rho_0 * j / v_e
    tmp2 = np.log( R ** 2 / rho_0 ** 2 + 1 )
    return tmp1 * tmp2

def eval_a_perp( m_e, q_e, M_i, Q_i, v_e, j, R ):
    rho_0 = np.abs( q_e * Q_i / ( m_e * v_e ** 2 ) )
    tmp1 = 2.0 * j * m_e * rho_0 / ( M_i**4 * q_e ) * np.pi**2 * R**2 * v_e
    tmp2 = 2.0 * M_i**3 * R    
    tmp3 = rho_0 * ( M_i * rho_0 )**(3.0/2.0) * np.arctan( R / rho_0 )
    return tmp1 * ( tmp2 - tmp3 )


def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("thermolization_[0-9]*.h5"):
        h5files.append( file )
    return h5files

def output_file_current_time( filename ):
    h5file = h5py.File( filename, mode = "r" )
    time = h5file["/Time_grid"].attrs["current_time"][0]
    h5file.close()
    return time


### Plot as points 

out_files = find_necessary_out_files()
out_files.sort( key = output_file_current_time )
print( out_files )

h5file = h5py.File( out_files[0], mode = "r" )
M_i, Q_i = get_ions_parameters( h5file )
m_e, q_e = get_electrons_parameters( h5file )
v_e = get_electrons_velocity( h5file )
j_e = get_electrons_current_density( h5file )
n_e = get_electrons_concentration( h5file )
R = get_electrons_beam_radius( h5file )
z_volume_size = h5file["/Spatial_mesh"].attrs["z_volume_size"][0]
x_volume_size = h5file["/Spatial_mesh"].attrs["x_volume_size"][0]

v_prll_initial = 0 # todo: read from config    
a_prll = eval_a_prll( m_e = m_e, q_e = q_e,
                      M_i = M_i, Q_i = Q_i,
                      v_e = v_e, j = j_e, R = R )

v_perp_initial = 0 # todo: read from config
a_perp = eval_a_perp( m_e = m_e, q_e = q_e,
                      M_i = M_i, Q_i = Q_i,
                      v_e = v_e, j = j_e, R = R )


### V_prll
plt.figure()
plt.xlabel( "V_prll [cm/s]" )
plt.ylabel( "Time" )

times = []
for filename in out_files:
    h5file = h5py.File( filename, mode = "r" )
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0] # delete
    p_z_num = h5file["/Particle_sources/ions/momentum_z"]
    v_prll = p_z_num / M_i

    times.append( h5file["/Time_grid"].attrs["current_time"][0] )    
    print( "times:", times )
    
    y = np.full_like( v_prll, times[-1] )
    plt.plot( v_prll, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )
    
    h5file.close()

v_prll_an = [ v_prll_initial + a_prll * t for t in times ]
plt.plot( v_prll_an, times, label = "theory", color = "b" )

plt.legend()
plt.yticks( times )
plt.savefig( "ion_velocities_as_points_prll.png" )


### V_perp

plt.figure()
plt.xlabel( "V_perp [cm/s]" )
plt.ylabel( "Time" )

times = []
for i, filename in enumerate(out_files):
    h5file = h5py.File( filename, mode = "r" )
    p_x_num = h5file["/Particle_sources/ions/momentum_x"]
    p_y_num = h5file["/Particle_sources/ions/momentum_y"]
    v_perp = np.sqrt( np.square( p_x_num ) + np.square( p_y_num ) ) / M_i

    times.append( h5file["/Time_grid"].attrs["current_time"][0] )
    
    y = np.full_like( v_perp, times[-1] )
    plt.plot( v_perp, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )

    # v_perp_mean = np.mean( v_perp )
    # v_perp_stdiv = np.std( v_perp )
    # plt.text( 3.1e5, time_step_num,
    #           "mean={:.2e}".format( v_perp_mean ),
    #           fontsize = 10 )
    # plt.text( 3.1e5, time_step_num - 0.3,
    #           "stdiv={:.2e}".format( v_perp_stdiv ),
    #           fontsize = 10 )
    #plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
    #plt.plot( z_an, r_an_lower, color = "g" )
    h5file.close()

v_perp_an = [ v_perp_initial + a_perp * t for t in times ]
plt.plot( v_perp_an, times, label = "theory", color = "b" )
print( v_perp_an )

plt.legend()
plt.yticks( times )
plt.savefig( "ion_velocities_as_points_perp.png" )



