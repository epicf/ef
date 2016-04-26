import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    outfile_name = "conducting_sphere_potential_fieldsWithoutParticles.h5"
    extract_sphere_properties_from_out_file( outfile_name )
    node_coords, phi = extract_full_nodecoords_and_potential_from_out_file( outfile_name )
    num_subset = num_subset_along_z( node_coords, phi )
    an = an_potential_at_num_points( num_subset )
    plot_comparision( num_subset, an )

    
def extract_sphere_properties_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r")
    sphere_object_hdf5_group = outfile['/Inner_regions/sphere_object']
    global sphere_center_x, sphere_center_y, sphere_center_z
    global sphere_radius, sphere_phi
    sphere_center_x = sphere_object_hdf5_group.attrs['origin_x']
    sphere_center_y = sphere_object_hdf5_group.attrs['origin_y']
    sphere_center_z = sphere_object_hdf5_group.attrs['origin_z']
    sphere_radius = sphere_object_hdf5_group.attrs['radius']
    sphere_phi = sphere_object_hdf5_group.attrs['potential']
    outfile.close()

    
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


def num_subset_along_z( node_coords, phi ):
    num_subset = []
    for ( (x,y,z), p ) in zip(node_coords, phi):
        if ( x == sphere_center_x and y == sphere_center_y ):
            num_subset.append( (x, y, z, p) )            
    num_subset = np.array( num_subset,
                           dtype=[('x', 'float'), ('y', 'float'),
                                  ('z', 'float'), ('phi', 'float')] )
    return( num_subset )
    

def an_potential_at_num_points( num_subset ):
    an = np.empty_like( num_subset )
    for ( i, (x, y, z) ) in enumerate( num_subset[['x','y','z']] ):
        phi = spherical_potential( x, y, z,
                                   sphere_center_x, sphere_center_y, sphere_center_z,
                                   sphere_radius, sphere_phi )
        an[i] = ( x, y, z, phi )
    return an


def spherical_potential( x, y, z,
                         center_x, center_y, center_z,
                         radius, phi_on_sphere ):
    dist = np.sqrt( (x - center_x) * (x - center_x) +
                    (y - center_y) * (y - center_y) +
                    (z - center_z) * (z - center_z) )
    if dist <= radius:
        potential = phi_on_sphere
    else:
        potential = phi_on_sphere * radius / dist
    return( potential )


def plot_comparision( num, an ):    
    plt.plot( num['z'], num['phi'], 
              linestyle='', marker='o',
              label = "Num" )
    plt.plot( an['z'], an['phi'],
              label = "An" )
    plt.legend()
    print( "Saving potential comparison to 'potential_along_z.png'")
    plt.savefig('potential_along_z.png')

main()


