from math import *

m = 9.8e-28
q = 4.8e-10
print( "q = {:.3e} [cgs]".format( q ) )
print( "m = {:.3e} [g]".format( m ) )

r_0 = 0.5
print( "beam_radius = {:.3e} [cm]".format( r_0 ) )

ampere_to_cgs = 2997924536.8431
I = 0.1 * ampere_to_cgs
print( "I = {:.3e} [A] = {:.3e} [cgs]".format( I / ampere_to_cgs, I ) )

ev_to_cgs = 1.60218e-12
E = 1000 * ev_to_cgs
v = sqrt( 2 * E / m )
print( "E = {:.3e} [eV] = {:.3e} [erg]".format( E / ev_to_cgs, E ) )
print( "v = {:.3e} [cm/s]".format( v ) )

r0_e_times_wider = e * r_0
z_e_times_wider = 1.46 * sqrt( m * v**3 / q / I ) * r_0
t_e_times_wider = 1.5 * sqrt( m * v / q / I ) * r_0
print( "r0_e_times_wider = {:.3e} [cm]".format( r0_e_times_wider ) )
print( "t_e_times_wider = {:.3e} [s]".format( t_e_times_wider ) )
print( "z_e_times_wider = {:.3e} [cm]".format( z_e_times_wider ) )

sim_time = 3.0e-9
n_of_steps = 100
dt = sim_time / n_of_steps
print( "simulation_time = {:.3e} [s]".format( sim_time ) )
print( "number_of_time_steps = {:d}".format( n_of_steps ) )
print( "time_step_size = {:.3e} [s]".format( dt ) )

num_of_real_particles = I * dt / q
print( "num_of_real_particles = {:.3e}".format( num_of_real_particles ) )

num_of_macro_particles = 5000
macro_q = I * dt / num_of_macro_particles
macro_m = macro_q / q * m
macro_mean_momentum = macro_m * v
print( "num_of_macro_particles = {:d}".format( num_of_macro_particles ) )
print( "macro_q = {:.3e} [cgs]".format( macro_q ) )
print( "macro_m = {:.3e} [g]".format( macro_m ) )
print( "macro_mean_momentum = {:.3e} [g * cm / s]".format( macro_mean_momentum ) )
