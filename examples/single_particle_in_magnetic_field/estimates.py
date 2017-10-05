from math import *

m = 9.8e-28
q = 4.8e-10
print( "q = {:.3e} [cgs]".format( q ) )
print( "m = {:.3e} [g]".format( m ) )

ev_to_cgs = 1.60218e-12
E_along = 1000 * ev_to_cgs
v_along = sqrt( 2 * E_along / m )
E_perp = 100 * ev_to_cgs
v_perp = sqrt( 2 * E_perp/2 / m )
print( "E_along = {:.3e} [eV] = {:.3e} [erg]".format( E_along / ev_to_cgs, E_along ) )
print( "E_perp = {:.3e} [eV] = {:.3e} [erg]".format( E_perp / ev_to_cgs, E_perp ) )
print( "v_along = {:.3e} [cm/s]".format( v_along ) )
print( "p_along = {:.3e} [g * cm/s]".format( v_along * m ) )
print( "v_perp = {:.3e} [cm/s]".format( v_perp ) )
print( "p_perp = {:.3e} [g * cm/s]".format( v_perp * m ) )
print( "" )

H = 1000
speed_of_light = 3.0e10
cyclotron_fr = q * H / m / speed_of_light
cyclotron_period = 2.0 * pi / cyclotron_fr
single_period_distance_along_field = v_along * cyclotron_period
larmor_r = m * speed_of_light * sqrt(2 * E_perp / m) / q / H
print( "H = {:.3e} [Gs]".format( H ) )
print( "c = {:.3e} [cm/s]".format( speed_of_light ) )
print( "Omega = {:.3e} [1/s]".format( cyclotron_fr ) )
print( "Cyclotron period = {:.3e} [s]".format( cyclotron_period ) )
print( "Single period distance along field= {:.3e} [cm]".format(
    single_period_distance_along_field ) )
print( "Larmor_r = {:.3e} [cm]".format( larmor_r ) )
print( "" )

z_distance = 5
t = z_distance / v_along
print( "z_distance = {:f} [cm]".format( z_distance ) )
print( "t = {:.3e} [s]".format( t ) )

sim_time = 3.0e-9
n_of_revolutions = sim_time / cyclotron_period
n_of_steps = 1000
dt = sim_time / n_of_steps
print( "simulation_time = {:.3e} [s]".format( sim_time ) )
print( "n_of_revolutions = {:.1f}".format( n_of_revolutions ) )
print( "number_of_time_steps = {:d}".format( n_of_steps ) )
print( "time_step_size = {:.3e} [s]".format( dt ) )
