import os, glob
import operator
import h5py
import numpy as np
import matplotlib.pyplot as plt

def get_time_potential_charge_absrbd_on_anode_from_h5( filename ):
    h5 = h5py.File( filename, mode="r")
    absorbed_charge = h5["/Inner_regions/anode"].attrs["total_absorbed_charge"][0]
    time = h5["/Time_grid"].attrs["current_time"][0]
    potential = h5["/Inner_regions/anode"].attrs["potential"][0]
    h5.close()
    # return( {"time": time, 
    #         "potential": potential, 
    #         "absorbed_charge": absorbed_charge } )
    return( (time, 
             potential, 
             absorbed_charge ) )
 

os.chdir("./")
# todo: remove hardcoding
prev_step_filename = "V*_*0900.h5"
last_step_filename = "V*_*1000.h5"
prev_step_vals = []
last_step_vals = []
for f in glob.glob( prev_step_filename ):
    prev_step_vals.append( get_time_potential_charge_absrbd_on_anode_from_h5( f ) )
for f in glob.glob( last_step_filename ):
    last_step_vals.append( get_time_potential_charge_absrbd_on_anode_from_h5( f ) )

prev_step_vals.sort( key = operator.itemgetter(1) )
last_step_vals.sort( key = operator.itemgetter(1) )

current = []
voltage = []
cgs_to_v = 300
for (t1,V1,q1), (t2,V2,q2) in zip( prev_step_vals, last_step_vals ):
    print( t2 - t1, V2 - V1, q2 - q1 )
    current.append( abs( ( q2 - q1 ) ) / ( t2 - t1 ) )
    voltage.append( V1 * cgs_to_v )

#print( current, voltage )
#A,B = np.polyfit( np.ln( current ), voltage, 1 )

plt.figure() 
axes = plt.gca()
axes.set_xlabel( "Voltage [V]" )
axes.set_ylabel( "Current [?]" )
#axes.set_xlim( [0, 1500] )
plt.plot( voltage, current,
          linestyle='', marker='o',
          label = "Num" )
#plt.plot( current_an, voltage_an,
#          label = "An" )
plt.legend()
plt.savefig('diode_VC.png')
