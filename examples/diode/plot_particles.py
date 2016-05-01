import os, sys, glob
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def main():
    h5files = find_necessary_out_files()
    extract_boundary_box_properties_from_out_file( h5files[0] )
    for f in h5files:
        h5 = h5py.File( f, driver = "core", mode = "r" )
        current_time = h5["/Time_grid"].attrs["current_time"][0]
        current_timestep_num = h5["/Time_grid"].attrs["current_node"][0]
        particles, mass = read_particles( h5 )
        plot_particles_3d( particles, mass, current_time, current_timestep_num )
        plot_particles_2d( particles, mass, current_time, current_timestep_num )
        h5.close()

def find_necessary_out_files():
    os.chdir("./")
    h5files = []
    for file in glob.glob("diode_[0-9]*.h5"):
        h5files.append( file )
    return h5files    

def read_particles( h5_file ):
    particles_h5 = h5_file["/Particle_sources/cathode_emitter"]
    particles = np.empty_like( particles_h5 )
    particles_h5.read_direct( particles )
    mass = particles_h5.attrs["mass"][0]
    return (particles, mass)

def plot_particles_3d( particles, mass, current_time, current_timestep_num ):
    cmap = plt.get_cmap( 'Oranges' )
    kin_energy = ( particles['momentum']['vec_x']**2 \
                   + particles['momentum']['vec_y']**2 \
                   + particles['momentum']['vec_z']**2 ) / ( 2 * mass )
    norm = mpl.colors.Normalize( 0, vmax = kin_energy.max() )
    colors = cmap( norm( kin_energy ) )
    colors[:,3] = 0.01 # set alpha
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.set_xlabel('X') 
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_xlim( 0, x_volume_size )
    ax.set_ylim( 0, y_volume_size )
    ax.set_zlim( 0, z_volume_size )
    ax.scatter( particles['position']['vec_x'],
                particles['position']['vec_y'],
                particles['position']['vec_z'],
                c = colors, edgecolors='none' )
    ax.view_init( azim = -30, elev = 40 )
    plt.title( "3d, Time={0:.3g}".format( current_time ), loc = "left" )
    #plt.show()
    plot_filename = "3d_{0:0=7d}.png".format( current_timestep_num )
    print( 'Saving 3d-particles-plot '
           'at t={0:.3g} to {1}'.format( current_time, plot_filename ) )
    plt.savefig( plot_filename )
    
def plot_particles_2d( particles, mass, current_time, current_timestep_num ):
    cmap = plt.get_cmap( 'Oranges' )
    kin_energy = ( particles['momentum']['vec_x']**2 \
                   + particles['momentum']['vec_y']**2 \
                   + particles['momentum']['vec_z']**2 ) / ( 2 * mass )
    norm = mpl.colors.Normalize( 0, vmax = kin_energy.max() )
    colors = cmap( norm( kin_energy ) )
    colors[:,3] = 0.01 # set alpha
    plt.figure(1)
    plt.subplot(121)
    # Side view (Z-Y plane)
    plt.xlim( 0, z_volume_size )
    plt.ylim( 0, y_volume_size )
    plt.scatter( particles['position']['vec_z'],
                 particles['position']['vec_y'],
                 c = colors, edgecolors='none',
                 marker='.' )
    plt.title( "Time={0:.3g}, Side view".format( current_time ) )
    plt.xlabel('Z') 
    plt.ylabel('Y') 
    # View along propagation direction (X-Y plane)
    plt.subplot(122)
    plt.xlim( 0, x_volume_size )
    plt.ylim( 0, y_volume_size )
    plt.scatter( particles['position']['vec_x'],
                 particles['position']['vec_y'],
                 c = colors, edgecolors='none',
                 marker='.' )
    plt.title( "Along propagation direction" )
    plt.xlabel('X') 
    plt.ylabel('Y') 
    #plt.show()
    plot_filename = "2d_{0:0=7d}.png".format( current_timestep_num )
    print( 'Saving 2d particles plot '
           'at t={0:.3g} to {1}'.format( current_time, plot_filename ) )
    plt.savefig( plot_filename )

def extract_boundary_box_properties_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r" )
    global x_cell_size, x_n_nodes, x_volume_size
    global y_cell_size, y_n_nodes, y_volume_size
    global z_cell_size, z_n_nodes, z_volume_size
    x_cell_size = outfile["/Spatial_mesh"].attrs["x_cell_size"][0]
    y_cell_size = outfile["/Spatial_mesh"].attrs["y_cell_size"][0]
    z_cell_size = outfile["/Spatial_mesh"].attrs["z_cell_size"][0]
    x_n_nodes = outfile["/Spatial_mesh"].attrs["x_n_nodes"][0]
    y_n_nodes = outfile["/Spatial_mesh"].attrs["y_n_nodes"][0]
    z_n_nodes = outfile["/Spatial_mesh"].attrs["z_n_nodes"][0]
    x_volume_size = outfile["/Spatial_mesh"].attrs["x_volume_size"][0]
    y_volume_size = outfile["/Spatial_mesh"].attrs["y_volume_size"][0]
    z_volume_size = outfile["/Spatial_mesh"].attrs["z_volume_size"][0]    
    outfile.close()
    
main()


