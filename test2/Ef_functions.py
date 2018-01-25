import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

def read_h5_potential(name_file):
    h5 = h5py.File( name_file , mode="r")
    pot=h5["/Spatial_mesh/potential"][:]    
    h5.close()
    return pot
    
def read_h5_particles(name_file):
    h5 = h5py.File( name_file , mode="r")
    particles = []
    for i in range(0,np.size(h5["/Particle_sources/cathode_emitter/position_x"][:])):
        particles.append(np.array([h5["/Particle_sources/cathode_emitter/position_x"][i],
                                           h5["/Particle_sources/cathode_emitter/position_y"][i],
                                           h5["/Particle_sources/cathode_emitter/position_z"][i],
#                                           h5["/Particle_sources/cathode_emitter/momentum_x"][i],
#                                           h5["/Particle_sources/cathode_emitter/momentum_y"][i],
                                           h5["/Particle_sources/cathode_emitter/momentum_z"][i],
                                           h5["/Particle_sources/cathode_emitter/particle_id"][i]]))
    h5.close()
    return particles

    
def read_h5_amount_of_particles(name_file):
    h5 = h5py.File( name_file , mode="r")
    volumesource=(len(h5["/Particle_sources/cathode_emitter/position_x"]))
    h5.close()
    return volumesource
    

def dens(name_file):
    h5 = h5py.File( name_file , mode="r")
    if ( len(h5["/Particle_sources/cathode_emitter"]) > 0 ):
        particles = np.array([h5["/Particle_sources/cathode_emitter/position_x"][:],
                                           h5["/Particle_sources/cathode_emitter/position_y"][:],
                                           h5["/Particle_sources/cathode_emitter/position_z"][:],
                                           h5["/Particle_sources/cathode_emitter/momentum_z"][:],
                                           h5["/Particle_sources/cathode_emitter/particle_id"][:],])
    
    h5.close()
    return
    
def read_h5_amount_of_particles_per_electrode(name_file,name_electrode):
    h5 = h5py.File( name_file , mode="r")
    volumesource=h5["/Inner_regions/"+name_electrode].attrs["total_absorbed_particles"][0]
    h5.close()
    return volumesource    

'''
name = glob.glob('*.h5')
name.sort()

cor = []
for i in range(1,1100000):
    cor.append([])

for i in name[1:151]:
    h5 = h5py.File( i , mode="r")  
    for j in range(0,np.size(h5["/Particle_sources/cathode_emitter/position_x"][:])):
        cor[h5["/Particle_sources/cathode_emitter/particle_id"][j]].append(np.array([h5["/Particle_sources/cathode_emitter/position_x"][j],
            h5["/Particle_sources/cathode_emitter/position_y"][j],
            h5["/Particle_sources/cathode_emitter/position_z"][j]]))
    h5.close()
#                                           h5["/Particle_sources/cathode_emitter/momentum_x"][i],
#                                           h5["/Particle_sources/cathode_emitter/momentum_y"][i],
#                                           h5["/Particle_sources/cathode_emitter/momentum_z"][i],
#                                           ))

plt.show()
#plt.ylim(0.198,0.203)
#plt.plot(num[2],num[1])
#plt.savefig('ref.png',dpi=2400)
'''
name = "out_test_0380000.h5"
#name.sort()
#npart = []
#nabs = []
#    nabs.append(read_h5_amount_of_particles_per_electrode(i,"cathode"))
#    par = read_h5_particles(i)

par = read_h5_particles("out_test_0010000.h5")
par = np.array(par)
par = np.transpose(par)

plt.plot(par[2],par[0],'.')

plt.show()
plt.xlabel("X axis, cm")
plt.ylabel("Y axis, cm")
plt.xlim(0.05,0.35)
plt.ylim(0.05,0.35)
plt.plot(par[0],par[1],'.')
plt.savefig("3D/xy/"+i+"xy"+".png", dpi = 900)
    
plt.show()
plt.xlabel("Z axis, cm")
plt.ylabel("Y axis, cm")
plt.xlim(0,2)
plt.ylim(0.05,0.35)
plt.plot(par[2],par[1],'.')
plt.savefig("3D/zy/"+i+"zy"+".png", dpi = 900)
    
plt.show()
plt.xlabel("Z axis, cm")
plt.ylabel("X axis, cm")
plt.xlim(0,2)
plt.ylim(0.05,0.35)
plt.plot(par[2],par[0],'.')
plt.savefig("3D/zx/"+i+"zx"+".png", dpi = 900)
    
plt.show()
plt.xlabel("Z axis, cm")
plt.ylabel("Velocity, cm/s")
plt.xlim(0,2)
plt.ylim(-4e+9,4e+9)
vz = []
vz = par[5] / 1.89e-25
plt.plot(par[2],vz,'.')
plt.savefig("zvz/"+i+"zvz"+".png", dpi = 900)



