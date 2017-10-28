import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import operator

ev_to_cgs = 1.60218e-12

def get_electrons_parameters( h5file ):
    mass = h5file["/Particle_sources/electrons"].attrs["mass"][0]
    charge = h5file["/Particle_sources/electrons"].attrs["charge"][0]
    return ( mass, charge )

def get_electrons_beam_radius( h5file ):
    return h5file["/Particle_sources/electrons"].attrs["cylinder_radius"][0]

def get_electrons_concentration( h5file ):
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

def eval_a_prll( m_e, q_e, M_i, Q_i, v_e, n_e, R ):
    rho_0 = np.abs( q_e * Q_i / ( m_e * v_e ** 2 ) )
    tmp1 = 2.0 * np.pi * m_e / M_i * n_e * rho_0**2 * v_e**2
    tmp2 = np.log( 1 + R ** 2 / rho_0 ** 2 )
    return tmp1 * tmp2

def eval_a_perp( m_e, q_e, M_i, Q_i, v_e, n_e, R ):
    rho_0 = np.abs( q_e * Q_i / ( m_e * v_e ** 2 ) )
    tmp1 = 2.0 * np.pi**2 * R**2 * m_e / M_i**4 * n_e * rho_0 * v_e**2
    tmp2 = 2.0 * M_i**3 * R    
    tmp3 = rho_0 * ( M_i * rho_0 )**(3.0/2.0) * np.arctan( R / rho_0 )
    return tmp1 * ( tmp2 - tmp3 )

def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("thermolization_[0-9]*.h5"):
        h5files.append( file )
    return h5files

def get_current_time_from_output_file( filename ):
    h5file = h5py.File( filename, mode = "r" )
    time = h5file["/Time_grid"].attrs["current_time"][0]
    h5file.close()
    return time


### Plot as points 

out_files = find_necessary_out_files()
out_files.sort( key = get_current_time_from_output_file )
print( out_files )

h5file = h5py.File( out_files[0], mode = "r" )
M_i, Q_i = get_ions_parameters( h5file )
m_e, q_e = get_electrons_parameters( h5file )
v_e = get_electrons_velocity( h5file )
n_e = get_electrons_concentration( h5file )
R = get_electrons_beam_radius( h5file )
z_volume_size = h5file["/Spatial_mesh"].attrs["z_volume_size"][0]
x_volume_size = h5file["/Spatial_mesh"].attrs["x_volume_size"][0]

v_prll_initial = 0 # todo: read from config    
a_prll = eval_a_prll( m_e = m_e, q_e = q_e,
                      M_i = M_i, Q_i = Q_i,
                      v_e = v_e, n_e = n_e, R = R )

v_perp_initial = 0 # todo: read from config
a_perp = np.abs( eval_a_perp( m_e = m_e, q_e = q_e,
                              M_i = M_i, Q_i = Q_i,
                              v_e = v_e, n_e = n_e, R = R ) )


### V_prll
plt.figure()
plt.xlabel( "V_prll [cm/s]" )
#plt.xlabel( "E_prll [eV]" )
plt.ylabel( "Time" )

times = []
for filename in out_files:
    h5file = h5py.File( filename, mode = "r" )
    p_z_num = h5file["/Particle_sources/ions/momentum_z"]
    v_prll = p_z_num / M_i
    #E_prll = M_i * v_prll**2 / 2 / ev_to_cgs

    current_time = h5file["/Time_grid"].attrs["current_time"][0] 
    times.append( current_time )    
    
    y = np.full_like( v_prll, current_time )
    plt.plot( v_prll, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )
    #plt.plot( E_prll, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )
    
    h5file.close()

test = 1
#test = 5e7
v_prll_an = [ v_prll_initial + a_prll * t * test for t in times ]
E_prll_an = [ M_i * v**2 / 2 / ev_to_cgs for v in v_prll_an ]
plt.plot( v_prll_an, times, label = "theory", color = "b" )
#plt.plot( E_prll_an, times, label = "theory", color = "b" )

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

    current_time = h5file["/Time_grid"].attrs["current_time"][0] 
    times.append( current_time )    
    
    y = np.full_like( v_perp, current_time )    
    plt.plot( v_perp, y, marker = '.', markersize = 10, color = "g", alpha = 0.005 )

    h5file.close()


test = 1
#test = 1e-12
v_perp_an = [ v_perp_initial + a_perp * t * test for t in times ]
plt.plot( v_perp_an, times, label = "theory", color = "b" )

plt.legend()
plt.yticks( times )
plt.savefig( "ion_velocities_as_points_perp.png" )



