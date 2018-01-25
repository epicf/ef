import h5py
import numpy as np

Bx_tmp = B[3].reshape(101,21,21)
By_tmp = B[4].reshape(101,21,21)
Bz_tmp = B[5].reshape(101,21,21)

Bx_tmp = Bx_tmp.transpose()
By_tmp = By_tmp.transpose()
Bz_tmp = Bz_tmp.transpose()

B_field = [[],[],[]]

for i in range(0,21):
    for j in range(0,21):
        for k in range(0,101):
            B_field[0].append(Bx_tmp[i][j][k])
            B_field[1].append(By_tmp[i][j][k])
            B_field[2].append(Bz_tmp[i][j][k])
    