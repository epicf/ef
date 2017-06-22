import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

filename = 'contour_0000100.h5'
h5file = h5py.File( filename, mode = "r" )

def get_source_current( h5file ):
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    charge = h5file["/Particle_sources/cathode_emitter"].attrs["charge"][0]
    particles_per_step = h5file[
        "/Particle_sources/cathode_emitter"].attrs["particles_to_generate_each_step"][0]
    current = particles_per_step * charge / time_step
    return current
    
def get_source_geometry( h5file ):
    axis_start_x = \
        h5file["/Particle_sources/cathode_emitter"].attrs["cylinder_axis_start_x"][0]
    axis_start_z = \
        h5file["/Particle_sources/cathode_emitter"].attrs["cylinder_axis_start_z"][0]
    radius = h5file["/Particle_sources/cathode_emitter"].attrs["cylinder_radius"][0]    
    return ( axis_start_x, axis_start_z, radius )

def get_source_particle_parameters( h5file ):
    mass = h5file["/Particle_sources/cathode_emitter"].attrs["mass"][0]
    charge = h5file["/Particle_sources/cathode_emitter"].attrs["charge"][0]
    momentum_z = h5file["/Particle_sources/cathode_emitter"].attrs["mean_momentum_z"][0]
    return ( mass, charge, momentum_z )

def beam_radius( u, r_0 ):
    return r_0 * np.exp( u ** 2 )

def beam_z( u, m, v, q, I, r_0 ):
    coeff = np.sqrt( m * v**3 / q / I ) * r_0
    subint = lambda t: np.exp( t * t )
    low_lim = 0
    up_lim = u
    integral_value = scipy.integrate.quad( subint, low_lim, up_lim )[0]
    return coeff * integral_value

beam_axis_x_pos, emitter_z_pos, r_0 = get_source_geometry( h5file )
I = get_source_current( h5file )
m, q, p = get_source_particle_parameters( h5file )
v = p / m

u_min = 0; u_max = 2; num_u_points = 100  # for u = 1, r = r(0) * 2.71812
u = np.linspace( u_min, u_max, num_u_points )
r_an = [ beam_radius( x, r_0 ) for x in u ]
r_an_upper = r_an + beam_axis_x_pos
r_an_lower = beam_axis_x_pos - r_an 
z_an = [ beam_z( x, m = m, v = v, q = q, I = I, r_0 = r_0 ) for x in u ]
z_an = z_an + emitter_z_pos

r_num = h5file["/Particle_sources/cathode_emitter/position_x"]
z_num = h5file["/Particle_sources/cathode_emitter/position_z"]

z_volume_size = h5file["/Spatial_mesh"].attrs["z_volume_size"][0]
x_volume_size = h5file["/Spatial_mesh"].attrs["x_volume_size"][0]
plt.xlabel( "Z [cm]" )
plt.ylabel( "X [cm]" )
plt.ylim( 0, x_volume_size )
plt.xlim( 0, z_volume_size )
plt.plot( z_num, r_num, '.', label = "num" )
plt.plot( z_an, r_an_upper, label = "theory", color = "g" )
plt.plot( z_an, r_an_lower, color = "g" )
plt.legend()
plt.savefig( "beam_contour.png" )
h5file.close()
