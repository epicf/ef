import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    outfile_name = "diode_fieldsWithoutParticles.h5"
    #extract_boundary_box_properties_from_out_file( outfile_name )
    node_coords, phi = extract_full_nodecoords_and_potential_from_out_file( outfile_name )
    num_subset = num_subset_middle_plane( node_coords, phi )
    plot_potential( num_subset )

    
def extract_boundary_box_properties_from_out_file( outfile_name ):
    print( "nothing" )

    
def extract_full_nodecoords_and_potential_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r" )
    node_coords_hdf5 = outfile['/Spatial_mesh/node_coordinates']
    num_potential_hdf5 = outfile['/Spatial_mesh/potential']
    node_coords = np.empty_like( node_coords_hdf5 )
    node_coords_hdf5.read_direct( node_coords )
    num_potential = np.empty_like( num_potential_hdf5 )
    num_potential_hdf5.read_direct( num_potential )
    outfile.close()
    return( node_coords, num_potential )


def num_subset_middle_plane( node_coords, phi ):    
    grid_y_size = 0.6 # todo: read from file
    num_subset = []
    for ( (x,y,z), p ) in zip(node_coords, phi):
        if ( y == grid_y_size / 2 ):
            num_subset.append( (x, y, z, p) )            
    num_subset = np.array( num_subset,
                           dtype=[('x', 'float'), ('y', 'float'),
                                  ('z', 'float'), ('phi', 'float')] )
    return( num_subset )
    

def plot_potential( num ):
    # todo: sort num before reshaping
    # todo: get size from out-file
    X = num['x'].reshape( ( 31,101 ) )
    Y = num['z'].reshape( ( 31,101 ) )
    Z = num['phi'].reshape( ( 31,101 ) )
    plt.figure()
    levels = np.linspace( Z.min(), Z.max(), 20 )
    CS = plt.contourf( X, Y, Z, levels )
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Potential, [CGS]')
    plt.xlabel( "X [cm]" )
    plt.ylabel( "Z [cm]" )
    plt.title('Potential at Y = y_size / 2')
    plt.show()    
    #print( "Saving potential comparison to 'potential_along_z.png'")
    #plt.savefig('potential_along_z.png')

main()


