import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    outfile_name = "diode_fieldsWithoutParticles.h5"
    extract_boundary_box_properties_from_out_file( outfile_name )
    node_coords, phi, el_field = extract_full_nodecoords_potential_fields_from_out_file( outfile_name )
    num_subset = num_subset_middle_plane( node_coords, phi, el_field )
    plot_potential( num_subset )
    plot_fields( num_subset )

    
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
    
def extract_full_nodecoords_potential_fields_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r" )
    node_coords_hdf5 = outfile['/Spatial_mesh/node_coordinates']
    num_potential_hdf5 = outfile['/Spatial_mesh/potential']
    el_field_hdf5 = outfile['/Spatial_mesh/electric_field']
    node_coords = np.empty_like( node_coords_hdf5 )
    num_potential = np.empty_like( num_potential_hdf5 )
    el_field = np.empty_like( el_field_hdf5 )
    node_coords_hdf5.read_direct( node_coords )
    num_potential_hdf5.read_direct( num_potential )
    el_field_hdf5.read_direct( el_field )
    outfile.close()
    return( node_coords, num_potential, el_field )


def num_subset_middle_plane( node_coords, phi, el_field ):
    plane_at_half_y = int( ( y_n_nodes - 1 ) / 2 ) * y_cell_size
    num_subset = []
    for ( (x,y,z), p, (ex,ey,ez) ) in zip(node_coords, phi, el_field):
        if ( y == plane_at_half_y ):
            num_subset.append( (x, y, z, p, ex, ey, ez) )            
    num_subset = np.array( num_subset,
                           dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'),
                                  ('phi', 'float'),
                                  ('Ex', 'float'), ('Ey', 'float'), ('Ez', 'float') ] )
    return( num_subset )
    

def plot_potential( num ):
    # todo: sort num before reshaping
    X = num['x'].reshape( ( x_n_nodes, z_n_nodes ) )
    Y = num['z'].reshape( ( x_n_nodes, z_n_nodes ) )
    Z = num['phi'].reshape( ( x_n_nodes, z_n_nodes ) )
    plt.figure()
    levels = np.linspace( Z.min(), Z.max(), 20 )
    CS = plt.contourf( X, Y, Z, levels )
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Potential, [CGS]')
    plt.xlabel( "X [cm]" )
    plt.ylabel( "Z [cm]" )
    plt.title('Potential at Y = y_volume_size / 2 = {0}'.format( int( ( y_n_nodes - 1 ) / 2 ) * y_cell_size ) )
    #plt.show()    
    print( "Saving potential plot to 'potential_fieldsWithoutParticles.png'")
    plt.savefig('potential_fieldsWithoutParticles.png')

def plot_fields( num ):
    plot_fields_quiver( num )
    plot_fields_streamplot( num )
    
def plot_fields_quiver( num ):
    # todo: sort num before reshaping
    X = num['x'].reshape( ( x_n_nodes, z_n_nodes ) )
    Y = num['z'].reshape( ( x_n_nodes, z_n_nodes ) )
    U = num['Ex'].reshape( ( x_n_nodes, z_n_nodes ) )
    V = num['Ez'].reshape( ( x_n_nodes, z_n_nodes ) )
    plt.figure()
    Q = plt.quiver( X, Y, U, V, angles='xy' )
    qk = plt.quiverkey(Q, 0.9, 0.95, 5, 'Ex-Ez',
                       labelpos='E',
                       coordinates='figure',
                       fontproperties={'weight': 'bold'})
    plt.xlabel( "X [cm]" )
    plt.ylabel( "Z [cm]" )
    plt.title('Ex, Ez at Y = y_volume_size / 2 = {0}'.format( int( ( y_n_nodes - 1 ) / 2 ) * y_cell_size ) )
    #plt.show()    
    print( "Saving electric field quiver plot to 'field_quiver_fieldsWithoutParticles.png'")
    plt.savefig('field_quiver_fieldsWithoutParticles.png')
    
def plot_fields_streamplot( num ):
    # todo: sort num before reshaping
    X = np.unique( num['x'] )
    Y = np.unique( num['z'] )
    U = np.transpose( num['Ex'].reshape( ( x_n_nodes, z_n_nodes ) ) )
    V = np.transpose( num['Ez'].reshape( ( x_n_nodes, z_n_nodes ) ) )
    plt.figure()
    plt.streamplot(X, Y, U, V, color='r')
    plt.xlabel( "X [cm]" )
    plt.ylabel( "Z [cm]" )
    plt.title('Ex, Ez at Y = y_volume_size / 2 = {0}'.format( int( ( y_n_nodes - 1 ) / 2 ) * y_cell_size ) )
    #plt.show()
    print( "Saving electric field streamplot to 'field_streamplot_fieldsWithoutParticles.png'")
    plt.savefig('field_streamplot_fieldsWithoutParticles.png')
    
main()


