from math import *

# estimates are done in SI units;
# to compose config file, conversion to CGS is provided 

kg_to_g = 1000
coulomb_to_statcoulomb = 2997924580
m = 9.8e-31
q = 1.6e-19
print( "q = {:.3e} [C] = {:.3e} [statC]".format( q, q * coulomb_to_statcoulomb ) )
print( "m = {:.3e} [kg] = {:.3e} [g]".format( m, m * kg_to_g ) )

m_to_cm = 100
x_0 = 0.001
y_width = 0.03
print( "beam_width = {:.3f} [m] = {:.3f} [cm]".format( x_0, x_0 * m_to_cm ) )
print( "y_size = {:.3f} [m] = {:.3f} [cm]".format( y_width, y_width * m_to_cm ) )

I = 0.1
linear_current_density = I / y_width
print( "I = {:.3f} [A]".format( I ) )
print( "linear_current_density = {:.3f} [A/m]".format( linear_current_density ) )

voltage = 1000
q_electron = 1.60218e-19
ev_to_joule = 1.60218e-19
E = voltage * q / q_electron * ev_to_joule
v = sqrt( 2 * E / m )
print( "U = {:.3f} [V]".format( voltage ) )
print( "E = {:.3f} [eV]".format( E / ev_to_joule ) )
print( "v = {:.3e} [m/s]".format( v ) )

eps_0 = 8.85e-12
p = linear_current_density / ( 4 * eps_0 * sqrt( 2 * q / m ) * voltage**1.5 )
x0_2_times_wider = 2 * x_0
z_2_times_wider = sqrt( 2 * x_0 / p )
t_2_times_wider = z_2_times_wider / v
print( "x0_2_times_wider = {:.3f} [m] = {:.3f} [cm]".format(
    x0_2_times_wider, x0_2_times_wider * m_to_cm ) )
print( "t_2_times_wider = {:.3e} [s]".format( t_2_times_wider ) )
print( "z_2_times_wider = {:.3e} [m] = {:.3e} [cm]".format(
    z_2_times_wider, z_2_times_wider * m_to_cm ) )

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
print( "macro_q = {:.3e} [C] = {:.3e} [statC]".format(
    macro_q, macro_q * coulomb_to_statcoulomb ) )
print( "macro_m = {:.3e} [kg] = {:.3e} [g]".format( macro_m, macro_m * kg_to_g ) )
print( "macro_mean_momentum = {:.3e} [kg * m / s] = {:.3e} [g * cm / s]".format(
    macro_mean_momentum, macro_mean_momentum * m_to_cm * kg_to_g ) )
