from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import h5py

def txt_to_h5(txt_name,h5_name):
    data = np.loadtxt(txt_name)
    h5file = h5py.File(h5_name,'w')
    h5file['data'] = data
    h5file.close()         

def get_aprox(h5_file_in,h5_gile_out):
    h5file = h5py.File(h5_file_in,'r')
    data = h5file["data"]
    return data[:,0:6]

txt_to_h5('field','B.h5')
x=get_aprox('B.h5','gg')

grid_x, grid_y, grid_z = np.mgrid[0:2:21j,0:2:21j,0:40:101j]

points = x[:,0:3]
values = [x[:,3],x[:,4],x[:,5]]
B_abs = np.sqrt(values[0]*values[0]+values[1]*values[1]+values[2]*values[2])

grid_xx = griddata(points, values[0], (grid_x, grid_y, grid_z), method='nearest')
grid_yy = griddata(points, values[1], (grid_x, grid_y, grid_z), method='nearest')
grid_zz = griddata(points, values[2], (grid_x, grid_y, grid_z), method='nearest')

grid_abs = griddata(points, B_abs , (grid_x, grid_y, grid_z), method='nearest')

plt.show()
plt.figure(figsize=(10,10))
#plt.imshow(grid_abs[:,20,:])
plt.streamplot(grid_z[:,10,:],grid_x[:,10,:],grid_zz[:,10,:],grid_xx[:,10,:], density=2.5)


B_field = [[],[],[]]

for i in range(0,21):
    for j in range(0,21):
        for k in range(0,101):
            B_field[0].append(grid_xx[i,j,k])
            B_field[1].append(grid_yy[i,j,k])
            B_field[2].append(grid_zz[i,j,k])

B_field=np.array(B_field)
    
f = h5py.File("out_test_0000000.h5","r+")

f.__delitem__("/Spatial_mesh/magnetic_field_x")
f.__delitem__("/Spatial_mesh/magnetic_field_y")
f.__delitem__("/Spatial_mesh/magnetic_field_z")
 
dsetx = f.create_dataset("/Spatial_mesh/magnetic_field_x", data = B_field[0]*10000) #T to cgs
dsety = f.create_dataset("/Spatial_mesh/magnetic_field_y", data = B_field[1]*10000)
dsetz = f.create_dataset("/Spatial_mesh/magnetic_field_z", data = B_field[2]*10000)

f.close()
            