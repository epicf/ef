from math import *

m = 9.8e-28
q = 4.8e-10
print( "q = {:.3e} [cgs]".format( q ) )
print( "m = {:.3e} [g]".format( m ) )

ev_to_cgs = 1.60218e-12
E = 1000 * ev_to_cgs
v = sqrt( 2 * E / m )
print( "E = {:.3e} [eV] = {:.3e} [erg]".format( E / ev_to_cgs, E ) )
print( "v = {:.3e} [cm/s]".format( v ) )
print( "p = {:.3e} [g * cm/s]".format( v * m ) )
print( "" )

H = 1000
speed_of_light = 3.0e10
cyclotron_fr = q * H / m / speed_of_light
cyclotron_period = 2.0 * pi / cyclotron_fr
single_period_distance = v * cyclotron_period
larmor_r = m * speed_of_light * v / q / H
print( "H = {:.3e} [Gs]".format( H ) )
print( "c = {:.3e} [cm/s]".format( speed_of_light ) )
print( "Omega = {:.3e} [1/s]".format( cyclotron_fr ) )
print( "Cyclotron period = {:.3e} [s]".format( cyclotron_period ) )
print( "Single period distance = {:.3e} [cm]".format( single_period_distance ) )
print( "Larmor_r = {:.3e} [cm]".format( larmor_r ) )
print( "" )

z_distance = 10
t = z_distance / v
print( "z_distance = {:f} [cm]".format( z_distance ) )
print( "t = {:.3e} [s]".format( t ) )

sim_time = 6.0e-9
n_of_steps = 1000
dt = sim_time / n_of_steps
print( "simulation_time = {:.3e} [s]".format( sim_time ) )
print( "number_of_time_steps = {:d}".format( n_of_steps ) )
print( "time_step_size = {:.3e} [s]".format( dt ) )

# num_of_macro_particles = 1
# macro_q = I * dt / num_of_macro_particles
# macro_m = macro_q / q * m
# macro_mean_momentum = macro_m * v
# print( "num_of_macro_particles = {:d}".format( num_of_macro_particles ) )
# print( "macro_q = {:.3e} [cgs]".format( macro_q ) )
# print( "macro_m = {:.3e} [g]".format( macro_m ) )
# print( "macro_mean_momentum = {:.3e} [g * cm / s]".format( macro_mean_momentum ) )
