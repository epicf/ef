import os, glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


### Numerical trajectory
num_trajectory = []
os.chdir("./")
h5files = []
for file in glob.glob("single_particle_mgn_field_[0-9]*.h5"):
    h5files.append( file )

h5files.sort()
for file in h5files:
    h5 = h5py.File( file, mode="r")
    t = h5["/Time_grid"].attrs["current_time"][0]
    if ( len(h5["/Particle_sources/emit_single_particle"]) > 0 ):
        particle = h5["/Particle_sources/emit_single_particle"][0]
        xyz = list( particle[1] )
        num_trajectory.append( [t] + xyz )
    h5.close()

# todo: add names to columns
num_trajectory = np.array( num_trajectory )

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

fig = plt.figure() 
ax = fig.gca( projection='3d' ) 
# for t, x, y, z in num_trajectory:     
#    ax.scatter(x, y, z, '.r-' )

x = num_trajectory[:,1]
y = num_trajectory[:,2]
z = num_trajectory[:,3]

# x = x[0:100]
# y = y[0:100]
# z = z[0:100]

ax.plot( x, y, z )

ax.set_xlabel('X Label') 
ax.set_ylabel('Y Label') 
ax.set_zlabel('Z Label')

plt.show()



### Analitical trajectory
h5 = h5py.File( "single_particle_mgn_field_0000000.h5", mode="r")

# Particle parameters; 10000 * C(6,12),
# m = 12 * 1.67e-24 * 10000 #[g],
# q = 4.8e-10 * 6 * 10000 #[cgs-charge]
m = h5["/Particle_sources/emit_single_particle"].attrs["mass"][0]
q = h5["/Particle_sources/emit_single_particle"].attrs["charge"][0]
# m = 1
# q = 1


# Magn field along z
#B0 = 1
B0 = h5["/External_magnetic_field"].attrs["external_magnetic_field_z"][0]

# Particle initial pos and velocity
#x0, y0, z0 = 5, 5, 1
#vx0, vy0, vz0 = 9.4e-19 / m, 9.4e-19 / m, 9.4e-19 / m,
#vx0, vy0, vz0 = 1, 1, 1
particle_initial_pos_and_vel = h5["/Particle_sources/emit_single_particle"][0]
x0, y0, z0 = particle_initial_pos_and_vel[1]
vx0, vy0, vz0 = list( particle_initial_pos_and_vel[2] ) / m

v_perp_len = np.sqrt( vx0**2 + vy0**2 )
v_prll = vz0

speed_of_light = 3e10
larmor_rad = m / abs(q) * v_perp_len / B0 * speed_of_light
larmor_freq = abs(q) / m * B0 / speed_of_light

def velocities(t):
    vx = vx0 * np.cos( larmor_freq * t ) + vy0 * np.sin( larmor_freq * t)
    vy = -vx0 * np.sin( larmor_freq * t ) + vy0 * np.cos( larmor_freq * t )
    vz = vz0
    return (vx, vy, vz)

def coords(t):
    # integrate( vx(t), dt )
    x = x0 + 1 / larmor_freq * ( vx0 * np.sin( larmor_freq * t ) - vy0 * np.cos( larmor_freq * t) + vy0 ) 
    y = y0 + 1 / larmor_freq * ( vx0 * np.cos( larmor_freq * t ) + vy0 * np.sin( larmor_freq * t) - vx0 ) 
    z = z0 + vz0 * t
    return ( x, y, z )

# 100 circles. 
simulation_time_100_circles = 100 * 2 * np.pi / larmor_freq
simulation_time_num = num_trajectory[-1][0]
simulation_time = simulation_time_num
#simulation_time = simulation_time_100_circles

print( m, q )
print( larmor_rad, 2 * np.pi / larmor_freq, simulation_time )

# npoints = 1000
# an_trajectory = np.empty( [npoints, 4] )
# for i in range( npoints ):
#     t = simulation_time * i / ( npoints - 1 )
#     x, y, z = coords( t )
#     an_trajectory[i] = [t, x, y ,z]

an_trajectory = np.empty_like( num_trajectory )
for i, t in enumerate( num_trajectory[:,0] ):
    x, y, z = coords( t )
    an_trajectory[i] = [t, x, y ,z]


from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

fig = plt.figure() 
ax = fig.gca( projection='3d' ) 
#for t, x, y, z in trajectory:     
#    ax.scatter(x, y, z, '.r-' )
x = an_trajectory[:,1]
y = an_trajectory[:,2]
z = an_trajectory[:,3]

#x = x[0:100]
#y = y[0:100]
#z = z[0:100]

ax.plot( x, y, z )

ax.set_xlabel('X Label') 
ax.set_ylabel('Y Label') 
ax.set_zlabel('Z Label')

plt.show()





#XY
# plt.figure(1)
# plt.subplot(311)
# plt.plot( trajectory[:,1], trajectory[:,2],
#     linestyle='', marker='o',
#     label = "Num" )

#XZ
# plt.subplot(312)
# plt.plot( trajectory[:,1], trajectory[:,3],
#     linestyle='', marker='o',
#     label = "Num" )

#YZ
# plt.subplot(313)
# plt.plot( trajectory[:,2], trajectory[:,3],
#     linestyle='', marker='o',
#     label = "Num" )

# plt.plot( z_coords_subset, analit_potential_subset,
#     label = "An" )
# plt.legend()
plt.savefig('trajectory.png')
    
