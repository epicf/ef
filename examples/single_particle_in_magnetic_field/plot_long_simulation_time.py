import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def main():
    num = extract_num_trajectory_from_out_files()
    an = eval_an_trajectory_at_num_time_points( num )
    plot_trajectories( num , an )

def extract_num_trajectory_from_out_files():
    out_files = find_necessary_out_files()    

    num_trajectory = []
    for f in out_files:
        num_trajectory.append( extract_time_pos_mom( f ) )    

    num_trajectory = remove_empty_and_sort_by_time( num_trajectory )
    num_trajectory = np.array( num_trajectory, 
                               dtype=[('t','float'),
                                      ('x','float'), ('y','float'), ('z','float'),
                                      ('px','float'), ('py','float'), ('pz','float') ] )
    return( num_trajectory )

def remove_empty_and_sort_by_time( num_trajectory ):
    removed_empty = [ x for x in num_trajectory if x ]
    sorted_by_time = sorted( removed_empty, key = lambda x: x[0] )
    return ( sorted_by_time )

def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("long_simulation_[0-9]*.h5"):
        h5files.append( file )
    return h5files

def extract_time_pos_mom( h5file ):
    h5 = h5py.File( h5file, mode="r")
    t = h5["/Time_grid"].attrs["current_time"][0]
    t_pos_mom = ()
    if ( len(h5["/Particle_sources/emit_single_particle"]) > 0 and
         len(h5["/Particle_sources/emit_single_particle/position_x"]) > 0 ):
        x = h5["/Particle_sources/emit_single_particle/position_x"][0]
        y = h5["/Particle_sources/emit_single_particle/position_y"][0]
        z = h5["/Particle_sources/emit_single_particle/position_z"][0]
        px = h5["/Particle_sources/emit_single_particle/momentum_x"][0]
        py = h5["/Particle_sources/emit_single_particle/momentum_y"][0]
        pz = h5["/Particle_sources/emit_single_particle/momentum_z"][0]
        t_pos_mom = (t, x, y, z, px, py, pz)
    h5.close()
    return( t_pos_mom )

def eval_an_trajectory_at_num_time_points( num_trajectory ):
    initial_out_file = "long_simulation_0000000.h5"
    initial_h5 = h5py.File( initial_out_file, mode="r")
    global B0, q, m
    B0 = extract_magn_field( initial_h5 )
    q, m = extract_particle_charge_and_mass( initial_h5 )
    global x0, y0, z0, px0, py0, pz0, vx0, vy0, vz0
    x0, y0, z0, px0, py0, pz0 = extract_initial_pos_and_mom( initial_h5 )
    vx0, vy0, vz0 = px0/m, py0/m, pz0/m 
    initial_h5.close()

    global v_perp_len, v_prll
    v_perp_len = np.sqrt( vx0**2 + vy0**2 )
    v_prll = vz0

    global speed_of_light, larmor_rad, larmor_freq
    speed_of_light = 3e10 # todo 
    larmor_rad = m / abs(q) * v_perp_len / B0 * speed_of_light
    larmor_freq = abs(q) / m * B0 / speed_of_light

    an_trajectory = np.empty_like( num_trajectory )
    for i, t in enumerate( num_trajectory['t'] ):
        x, y, z = coords( t )
        vx, vy, vz = velocities(t)
        px, py, pz = vx * m, vy * m, vz * m
        an_trajectory[i] = ( t, x, y ,z, px, py, pz )

    return( an_trajectory )
    
def extract_magn_field( h5 ):
    B0 = h5["/External_fields/mgn_uni"].attrs["magnetic_uniform_field_z"][0]
    return B0

def extract_particle_charge_and_mass( h5 ):
    q = h5["/Particle_sources/emit_single_particle"].attrs["charge"][0]
    m = h5["/Particle_sources/emit_single_particle"].attrs["mass"][0]
    return (q, m)

def extract_initial_pos_and_mom( h5 ):
    x0 = h5["/Particle_sources/emit_single_particle/position_x"][0]
    y0 = h5["/Particle_sources/emit_single_particle/position_y"][0]
    z0 = h5["/Particle_sources/emit_single_particle/position_z"][0]
    px0 = h5["/Particle_sources/emit_single_particle/momentum_x"][0]
    py0 = h5["/Particle_sources/emit_single_particle/momentum_y"][0]
    pz0 = h5["/Particle_sources/emit_single_particle/momentum_z"][0]
    return( x0, y0, z0, px0, py0, pz0 )

def velocities(t):
    sign_larmor_freq = larmor_freq * np.sign( q )
    vx = vx0 * np.cos( sign_larmor_freq * t ) + vy0 * np.sin( sign_larmor_freq * t )
    vy = -vx0 * np.sin( sign_larmor_freq * t ) + vy0 * np.cos( sign_larmor_freq * t )
    vz = vz0
    return ( vx, vy, vz )

def coords(t):
    sign_larmor_freq = larmor_freq * np.sign( q )
    x = x0 + 1 / sign_larmor_freq * ( vx0 * np.sin( sign_larmor_freq * t ) - 
                                      vy0 * np.cos( sign_larmor_freq * t ) + vy0 )
    y = y0 + 1 / sign_larmor_freq * ( vx0 * np.cos( sign_larmor_freq * t ) +
                                      vy0 * np.sin( sign_larmor_freq * t ) - vx0 )
    z = z0 + vz0 * t
    return ( x, y, z )


def plot_trajectories( num , an ):
    plot_3d( num, an )
    plot_2d( num, an )
    plot_kin_en( num , an )

def plot_3d( num, an ):
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.plot( an['x'], an['y'], an['z'], 'g-', linewidth = 3, label = "An" )
    ax.plot( num['x'][::2], num['y'][::2], num['z'][::2], 'b.',
             markersize = 6, label = "Num" )
    ax.set_xlabel('X [cm]') 
    ax.set_ylabel('Y [cm]') 
    ax.set_zlabel('Z [cm]')
    plt.legend( loc = 'upper left', title="3d" )
    #plt.show()
    print( 'Saving 3d trajectory plot to "long_sim_3d.png"' )
    plt.savefig('long_sim_3d.png')
    
def plot_2d( num, an ):
    plt.figure( figsize=( 16, 6 ) )
    plt.subplots_adjust( left = None, bottom = None,
                         right = None, top = None,
                         wspace = 0.4, hspace = None )
    #XY
    ax = plt.subplot(131)
    plt.plot( num['x'], num['y'],
              linestyle='', marker='o',
              label = "Num" )
    plt.plot( an['x'], an['y'],
              linestyle='', marker='.', lw = 3,
              label = "An" )
    ax.set_xlabel('X [cm]') 
    ax.set_ylabel('Y [cm]') 
    plt.legend( loc = 'upper left', title="XY", bbox_to_anchor=(-0.6,1) )
    #ZX
    ax = plt.subplot(132)
    plt.plot( num['z'], num['x'],
              linestyle='', marker='o',
              label = "Num" )
    plt.plot( an['z'], an['x'],
              linestyle='-', marker='', lw = 3,
              label = "An" )
    ax.set_xlabel('Z [cm]') 
    ax.set_ylabel('X [cm]')
    ax.text(0.05, 0.92, 'ZX',
            transform=ax.transAxes, fontsize=15)
    #plt.legend( loc = 'upper left', title="ZX" )
    #ZY
    ax = plt.subplot(133)
    plt.plot( num['z'], num['y'],
              linestyle='', marker='o',
              label = "Num" )
    plt.plot( an['z'], an['y'],
              linestyle='-', marker='', lw = 3,
              label = "An" )
    ax.set_xlabel('Z [cm]') 
    ax.set_ylabel('Y [cm]')
    ax.text(0.88, 0.92, 'ZY',
            transform=ax.transAxes, fontsize=15)
    #plt.legend( loc = 'upper left', title="ZY" )
    print( 'Saving 2d trajectory projection plots to "long_sim_2d.png"' )
    plt.savefig('long_sim_2d.png')

    
def plot_kin_en( num , an ):
    E_num = ( num['px']**2 + num['py']**2 + num['pz']**2 ) / ( 2 * m )
    E_an = ( an['px']**2 + an['py']**2 + an['pz']**2 ) / ( 2 * m )
    t = num['t']
    plt.figure()
    axes = plt.gca()
    axes.set_xlabel( "t [s]" )
    axes.set_ylabel( "E [erg]" )
    # axes.set_ylim( [min( E_an.min(), E_num.min() ),
    #                 max( E_an.max(), E_num.max() ) ] )
    line, = plt.plot( t, E_num, 'o' )
    line.set_label( "Num" )
    line, = plt.plot( t, E_an, ls = 'solid', lw = 3 )
    line.set_label( "An" )
    plt.legend( loc = 'upper right' )
    print( 'Saving kinetic energy comparison plot to "long_sim_kin_en.png"' )
    plt.savefig( 'long_sim_kin_en.png' )


main()
