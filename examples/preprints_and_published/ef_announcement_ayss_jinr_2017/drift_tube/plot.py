import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_source_parameters(h5file):
    radius_beam = h5file["/Particle_sources/cathode_emitter"].attrs["cylinder_radius"][0]
    charge = h5file["/Particle_sources/cathode_emitter"].attrs["charge"][0]
    mass = h5file["/Particle_sources/cathode_emitter"].attrs["mass"][0]
    momentum_z = h5file["/Particle_sources/cathode_emitter"].attrs["mean_momentum_z"][0]
    particle_per_step = h5file["/Particle_sources/cathode_emitter"].attrs["particles_to_generate_each_step"][0]
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    radius_tube = h5file["/Inner_regions/tube"].attrs["inner_radius"][0]
    return radius_beam,charge,mass,momentum_z,particle_per_step,time_step,radius_tube

def get_mesh_parameters(h5file):
    x_coord = h5file["/Spatial_mesh/node_coordinates_x"][:]
    y_coord = h5file["/Spatial_mesh/node_coordinates_y"][:]
    z_coord = h5file["/Spatial_mesh/node_coordinates_z"][:]
    pot=h5file["/Spatial_mesh/potential"][:]
    return [x_coord, y_coord, z_coord, pot]

def get_beam_parameters(h5file):
    beam_center_x = h5file["/Particle_sources/cathode_emitter"].attrs["cylinder_axis_end_x"][0]
    beam_middle_z = h5file["/Spatial_mesh/node_coordinates_z"][np.size(h5file["/Spatial_mesh/node_coordinates_z"][:]) / 2]
    return beam_center_x,beam_middle_z

def evaluate_velocity_current_square(radius_beam,charge,mass,momentum_z,particle_per_step,time_step):
    ampere_to_cgs = 2997924536.8431
    velocity = momentum_z / mass
    current = particle_per_step * charge / time_step/ampere_to_cgs
    square = radius_beam**2*np.pi
    return velocity,current,square

def evaluate_k_const(velocity,current,square):
    k_const = 2.85e+12*current/velocity/square
    return k_const

sgse_to_volts = 300

filename = "pot_0000101.h5"
h5file = h5py.File( filename, mode="r")

pot = np.transpose(get_mesh_parameters(h5file))
beam_center_x,beam_middle_z = get_beam_parameters(h5file)
radius_beam,charge,mass,momentum_z,particle_per_step,time_step,radius_tube = get_source_parameters(h5file)
velocity,current,square = evaluate_velocity_current_square(radius_beam,charge,mass,momentum_z,particle_per_step,time_step)
k_const = evaluate_k_const(velocity,current,square)


R_beam = np.linspace(0,radius_beam,100)
R_empty = np.linspace(radius_beam,radius_tube,100)
phi_beam = -k_const*(R_beam**2-radius_beam**2*(1-2*np.log(radius_beam/radius_tube)))
phi_empty = -2*k_const*radius_beam**2*np.log(R_empty/radius_tube)

pot = pot[np.where(pot[:, 0] == beam_center_x)]
pot = pot[np.where(pot[:, 2] == beam_middle_z)]
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel("X coordinate, mm", fontsize=18 )
plt.ylabel("Potential, V", fontsize=18 )
plt.xlim( 0, 4.5 )
cm_to_mm = 10
plt.plot( (R_beam + beam_center_x) * cm_to_mm, phi_beam,
          color = 'r', lw = 4,
          label = "Theory, $r \leq r_{beam}$" )
plt.plot( (-1 * R_beam + beam_center_x) * cm_to_mm, phi_beam,
          color = 'r', lw = 4,
          label='_nolegend_' )
plt.plot( (R_empty + beam_center_x) * cm_to_mm, phi_empty,
          color = 'g', lw = 4, ls = '--', dashes=(7, 3),
          label = "Theory, $r > r_{beam}$" )
plt.plot( (-1 * R_empty + beam_center_x) * cm_to_mm, phi_empty,
          color = 'g', lw = 4, ls = '--', dashes=(7, 3),
          label='_nolegend_' )
plt.plot( pot[10:-10, 1] * cm_to_mm, pot[10:-10, 3] * sgse_to_volts,
          '.', color = "blue", label = "Simulation",
          mew = 4, ms = 4 )
plt.legend( loc='lower right', borderaxespad=0., fontsize=18 )
plt.savefig( "pot.png" )
h5file.close()



