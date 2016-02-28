import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

trajectory = []
os.chdir("./")
for file in glob.glob("single_particle_mgn_field_[0-9]*.h5"):
    h5 = h5py.File( file, mode="r")
    t = h5["/Time_grid"].attrs["current_time"][0]
    particle = h5["/Particle_sources/emit_single_particle"][0]
    xyz = list( particle[1] )
    trajectory.append( [t] + xyz )
    h5.close()

# todo: add names to columns
trajectory = np.array( trajectory )

#XY
plt.figure(1)
plt.subplot(311)
plt.plot( trajectory[:,1], trajectory[:,2],
    linestyle='', marker='o',
    label = "Num" )

#XZ
plt.subplot(312)
plt.plot( trajectory[:,1], trajectory[:,3],
    linestyle='', marker='o',
    label = "Num" )

#YZ
plt.subplot(313)
plt.plot( trajectory[:,2], trajectory[:,3],
    linestyle='', marker='o',
    label = "Num" )

# plt.plot( z_coords_subset, analit_potential_subset,
#     label = "An" )
# plt.legend()
plt.savefig('trajectory.png')
    
