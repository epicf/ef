import h5py
import numpy as np
import matplotlib.pyplot as plt

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

calculation_results = h5py.File( 
    "conducting_sphere_potential_fieldsWithoutParticles.h5", 
    driver="core", 
    mode="r")

node_coords_hdf5 = calculation_results['/Spatial_mesh/node_coordinates']
node_coords = np.empty_like( node_coords_hdf5 )
node_coords_hdf5.read_direct( node_coords )

num_potential_hdf5 = calculation_results['/Spatial_mesh/potential']
num_potential = np.empty_like( num_potential_hdf5 )
num_potential_hdf5.read_direct( num_potential )

sphere1_object_hdf5_group = calculation_results['/Inner_regions/sphere_object1']
sphere1_center_x = sphere1_object_hdf5_group.attrs['origin_x']
sphere1_center_y = sphere1_object_hdf5_group.attrs['origin_y']
sphere1_center_z = sphere1_object_hdf5_group.attrs['origin_z']
sphere1_radius = sphere1_object_hdf5_group.attrs['radius']
sphere1_phi = sphere1_object_hdf5_group.attrs['potential']
sphere1_potential = lambda x,y,z: spherical_potential( x, y, z,
                                                       sphere1_center_x,
                                                       sphere1_center_y,
                                                       sphere1_center_z,
                                                       sphere1_radius,
                                                       sphere1_phi )
                                                              

sphere2_object_hdf5_group = calculation_results['/Inner_regions/sphere_object2']
sphere2_center_x = sphere2_object_hdf5_group.attrs['origin_x']
sphere2_center_y = sphere2_object_hdf5_group.attrs['origin_y']
sphere2_center_z = sphere2_object_hdf5_group.attrs['origin_z']
sphere2_radius = sphere2_object_hdf5_group.attrs['radius']
sphere2_phi = sphere2_object_hdf5_group.attrs['potential']
sphere2_potential = lambda x,y,z: spherical_potential( x, y, z,
                                                       sphere2_center_x,
                                                       sphere2_center_y,
                                                       sphere2_center_z,
                                                       sphere2_radius,
                                                       sphere2_phi )
 


### Subset points.
line_along_z_idx = np.empty( len(node_coords), dtype = int )
current = 0
for idx, (x, y, z) in enumerate( node_coords ):
    if ( x == 5.0 and y == 5.0 ):
        line_along_z_idx[current] = idx
        current = current + 1
line_along_z_idx = np.resize( line_along_z_idx, current )


z_coords_subset = np.empty( len(line_along_z_idx) )
for i, idx in enumerate( line_along_z_idx ):
    z_coords_subset[i] = node_coords[idx][2]
num_potential_subset = num_potential[ line_along_z_idx ]
analit_potential_subset = np.empty( len( line_along_z_idx ) )
for i, idx in enumerate( line_along_z_idx ):
    x, y, z = node_coords[idx]
    analit_potential_subset[i] = sphere1_potential( x, y, z ) + \
                                 sphere2_potential( x, y, z ) 


plt.plot( z_coords_subset, num_potential_subset, 
    linestyle='', marker='o',
    label = "Num" )
plt.plot( z_coords_subset, analit_potential_subset,
    label = "An" )
plt.legend()
plt.savefig('potential_along_z.png')
#plot( line_along_z$vec_z, line_along_z$phi )
#lines( line_along_z$vec_z, line_along_z$an, add=T )
