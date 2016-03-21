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

sphere_object_hdf5_group = calculation_results['/Inner_regions/sphere_object']
sphere_center_x = sphere_object_hdf5_group.attrs['origin_x']
sphere_center_y = sphere_object_hdf5_group.attrs['origin_y']
sphere_center_z = sphere_object_hdf5_group.attrs['origin_z']
sphere_radius = sphere_object_hdf5_group.attrs['radius']
sphere_phi = sphere_object_hdf5_group.attrs['potential']
current_sphere_potential = lambda x,y,z: spherical_potential( x, y, z,
                                                              sphere_center_x,
                                                              sphere_center_y,
                                                              sphere_center_z,
                                                              sphere_radius,
                                                              sphere_phi )
                                                              


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
    analit_potential_subset[i] = current_sphere_potential( x, y, z ) 

plt.plot( z_coords_subset, num_potential_subset, 
    linestyle='', marker='o',
    label = "Num" )
plt.plot( z_coords_subset, analit_potential_subset,
    label = "An" )
plt.legend()
plt.savefig('potential_along_z.png')
#plot( line_along_z$vec_z, line_along_z$phi )
#lines( line_along_z$vec_z, line_along_z$an, add=T )
