import numpy as np

m = 9.8e-28
q = 4.8e-10
print( "q = {:.3e} [cgs]".format( q ) )
print( "m = {:.3e} [g]".format( m ) )
print("")

r_0 = 1.0
print( "beam_radius = {:.3e} [cm]".format( r_0 ) )

# ampere_to_cgs = 2997924536.8431
# I = 0.1 * ampere_to_cgs
# print( "I = {:.3e} [A] = {:.3e} [cgs]".format( I / ampere_to_cgs, I ) )

ev_to_cgs = 1.60218e-12
E = 1000 * ev_to_cgs
v = np.sqrt( 2 * E / m )
print( "E = {:.3f} [eV] = {:.3e} [erg]".format( E / ev_to_cgs, E ) )
print( "v = {:.3e} [cm/s]".format( v ) )

print("")
sim_time = 3.0e-9
n_of_steps = 100
dt = sim_time / n_of_steps
print( "simulation_time = {:.3e} [s]".format( sim_time ) )
print( "number_of_time_steps = {:d}".format( n_of_steps ) )
print( "time_step_size = {:.3e} [s]".format( dt ) )

print("")
electrons_path = dt * v
print( "electrons_path_per_step = {:.3e} [cm] = {:.3f} [mm]".format( electrons_path,
    electrons_path * 10 ) )

M = 1.672e-24
Q = 4.8e-10
cyl_len = 6
beam_vol = np.pi * r_0**2 * cyl_len
N_electrons = 1e7
n_e = N_electrons / beam_vol

def eval_a_prll( m_e, q_e, M_i, Q_i, v_e, n_e, R ):
    rho_0 = np.abs( q_e * Q_i / ( m_e * v_e ** 2 ) )
    tmp1 = 2.0 * np.pi * m_e / M_i * n_e * rho_0**2 * v_e**2
    tmp2 = np.log( R ** 2 / rho_0 ** 2 + 1 )
    return tmp1 * tmp2

print("")
print( "num_of_electrons = {:.3e}".format( N_electrons ) )
print( "n_e = {:.3e}".format( n_e ) )

print("")
a_prll = eval_a_prll( m_e = m, q_e = q,
                      M_i = M, Q_i = Q,
                      v_e = v, n_e = n_e, R = r_0 )
print( "a_prll = {:.3e}".format( a_prll ) )

expected_E_ions = 0.0001 * ev_to_cgs
v_ions = np.sqrt( 2 * expected_E_ions / M )
expected_time = np.sqrt( 2.0 * expected_E_ions / ( M * a_prll**2 ) )
print( "expected E ions = {:.3e} [eV] = {:.3e} [erg]".format(
    expected_E_ions / ev_to_cgs, expected_E_ions ) )
print( "v = {:.3e} [cm/s]".format( v_ions ) )
print( "expected time = {:.3e} [s]".format( expected_time ) )


# num_of_macro_particles = 10000
# macro_q = I * dt / num_of_macro_particles
# macro_m = macro_q / q * m
# macro_mean_momentum = macro_m * v
# print( "num_of_macro_particles = {:d}".format( num_of_macro_particles ) )
# print( "macro_q = {:.3e} [cgs]".format( macro_q ) )
# print( "macro_m = {:.3e} [g]".format( macro_m ) )
# print( "macro_mean_momentum = {:.3e} [g * cm / s]".format( macro_mean_momentum ) )

print("")

