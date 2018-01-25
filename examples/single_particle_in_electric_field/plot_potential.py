import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    outfile_name = "single_particle_electric_field_fieldsWithoutParticles.h5"
    #extract_boundary_box_properties_from_out_file( outfile_name )
    node_x, node_y, node_z, phi = extract_full_nodecoords_and_potential_from_out_file( outfile_name )
    num_subset = num_subset_middle_plane( node_x, node_y, node_z, phi )
    plot_potential( num_subset )

    
def extract_boundary_box_properties_from_out_file( outfile_name ):
    print( "nothing" )


def extract_full_nodecoords_and_potential_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r" )
    node_coords_x_hdf5 = outfile['/Spatial_mesh/node_coordinates_x']
    node_coords_y_hdf5 = outfile['/Spatial_mesh/node_coordinates_y']
    node_coords_z_hdf5 = outfile['/Spatial_mesh/node_coordinates_z']
    num_potential_hdf5 = outfile['/Spatial_mesh/potential']
    node_coords_x = np.empty_like( node_coords_x_hdf5 )
    node_coords_y = np.empty_like( node_coords_y_hdf5 )
    node_coords_z = np.empty_like( node_coords_z_hdf5 )
    num_potential = np.empty_like( num_potential_hdf5 )
    node_coords_x_hdf5.read_direct( node_coords_x )
    node_coords_y_hdf5.read_direct( node_coords_y )
    node_coords_z_hdf5.read_direct( node_coords_z )
    num_potential_hdf5.read_direct( num_potential )
    outfile.close()
    return( node_coords_x, node_coords_y, node_coords_z, num_potential )

def num_subset_middle_plane( node_x, node_y, node_z, phi ):    
    grid_y_size = 10.0 # todo: read from file
    num_subset = []
    for ( x, y, z, p ) in zip( node_x, node_y, node_z, phi ):
        if ( y == grid_y_size / 2 ):
            num_subset.append( (x, y, z, p) )            
    num_subset = np.array( num_subset,
                           dtype=[('x', 'float'), ('y', 'float'),
                                  ('z', 'float'), ('phi', 'float')] )
    return( num_subset )
    

def plot_potential( num ):
    # todo: sort num before reshaping
    # todo: get size from out-file
    X = num['x'].reshape( ( 41,101 ) ) # 41 node with varying X and fixed Z, then next X-row with different Z. 
    Y = num['z'].reshape( ( 41,101 ) )
    Z = num['phi'].reshape( ( 41,101 ) )
    plt.figure()
    levels = np.linspace( Z.min(), Z.max(), 12 )
    CS = plt.contourf( X, Y, Z, levels )
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Potential, [CGS]')
    plt.xlabel( "X [cm]" )
    plt.ylabel( "Z [cm]" )
    plt.title('Potential at Y = y_size / 2')
    #plt.show()    
    print( "Saving potential comparison to 'potential_along_z.png'")
    plt.savefig('potential_along_z.png')

main()


