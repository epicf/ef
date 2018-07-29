# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:19:56 2017
Example for contour of ribbon electron beam
@author: Boytsov
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

SGSE_conv_unit_current_to_A = 3e10 * 0.1;     #from current units SGSE to A
SI_conv_cm_to_m = 0.01;      
SI_conv_g_to_kg = 0.001
SI_conv_Fr_to_C = 3.3356409519815207e-10

def get_source_current( h5file ):
    time_step = h5file["/Time_grid"].attrs["time_step_size"][0]
    charge = h5file["/Particle_sources/cathode_emitter"].attrs["charge"][0]
    particles_per_step = h5file[
        "/Particle_sources/cathode_emitter"].attrs["particles_to_generate_each_step"][0]
    current = np.abs(particles_per_step * charge / time_step)
    return current / SGSE_conv_unit_current_to_A 
    
def get_source_particle_parameters( h5file ):
    mass = h5file["/Particle_sources/cathode_emitter"].attrs["mass"][0]
    charge = h5file["/Particle_sources/cathode_emitter"].attrs["charge"][0]
    momentum_z = h5file["/Particle_sources/cathode_emitter"].attrs["mean_momentum_z"][0]
    return ( mass * SI_conv_g_to_kg, 
            charge * SI_conv_Fr_to_C, 
            momentum_z * SI_conv_g_to_kg * SI_conv_cm_to_m )
    
def get_source_geometry( h5file ):
    start_y = h5["/Particle_sources/cathode_emitter"].attrs["box_y_top"][0]
    end_y = h5["/Particle_sources/cathode_emitter"].attrs["box_y_bottom"][0]
    start_x = h5["/Particle_sources/cathode_emitter"].attrs["box_x_left"][0]
    end_x = h5["/Particle_sources/cathode_emitter"].attrs["box_x_right"][0]
    length_of_cathode = start_y-end_y
    half_thick_of_cathode = (start_x-end_x) / 2
    center_of_beam = (start_x+end_x) / 2    
    return ( length_of_cathode * SI_conv_cm_to_m, 
            half_thick_of_cathode * SI_conv_cm_to_m, 
            center_of_beam * SI_conv_cm_to_m )
    
def get_zlim( h5file ):
    start_z = h5["/Particle_sources/cathode_emitter"].attrs["box_z_near"][0]
    end_z = h5["/Spatial_mesh/"].attrs["z_volume_size"][0]
    return( start_z * SI_conv_cm_to_m, 
           end_z * SI_conv_cm_to_m)

def get_voltage( momentum_z, mass, charge ):
    energy = (momentum_z * momentum_z) / (2 * mass)
    voltage = energy / np.abs(charge)
    return voltage 
    
def get_current_dens(current,length_of_cathode):
    current_dens = current / length_of_cathode 
    return current_dens

def p_const( linear_current_density , voltage, charge , mass):                                     
    eps = 8.85e-12
    p_const = 1 / (4*eps*(np.abs(2*charge/mass))**0.5) * linear_current_density / voltage**1.5 ;   
    return p_const                                           


def contour( z_position , half_thick , angle , p_const):
    contour = half_thick + np.tan(angle) * z_position + p_const / 2 * (z_position * z_position)         
    return contour    

filename = 'contour_0001000.h5'
h5 = h5py.File( filename, mode="r")

current = get_source_current( h5 )
mass, charge, momentum_z = get_source_particle_parameters( h5 )
length_of_cathode, half_width, center_of_beam = get_source_geometry( h5 )
start_z, end_z = get_zlim( h5 )
voltage = get_voltage( momentum_z, mass, charge )
current_dens = get_current_dens(current,length_of_cathode)

conv_deg_to_rad = np.pi/180
angle = 0 * conv_deg_to_rad          #angle of beam
steps_z = 100
position_z = np.arange(start_z,end_z,(end_z-start_z)/steps_z)                  # points in z direction, from 0 to 0.01 m with step 0,00001 m 

p_cons = p_const(current_dens,voltage,charge,mass)                             # constant from equation of motion calculation 
contour = contour( position_z , half_width , angle , p_cons)                   # countour calculation, m

h5 = h5py.File( filename , mode="r") # read h5 file
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel("Z position, mm",fontsize=18)
plt.ylabel("X position, mm",fontsize=18)

plt.plot(h5["/Particle_sources/cathode_emitter/position_z"][:]*SI_conv_cm_to_m*1000, 
             (h5["/Particle_sources/cathode_emitter/position_x"][:]*SI_conv_cm_to_m - center_of_beam)*1000,
             '.',label="Calculated Points") #plot particles
plt.plot(position_z*1000,contour*1000,ms=4,color='r',lw=4,label="Analytic Curve")
plt.plot(position_z*1000,(-1)*contour*1000,ms=4,color='r',lw=4)
# plot countour in cm and move to left z of beam and top x of beam neat cathode
leg=plt.legend( loc=0, borderaxespad=0.,fontsize=18, markerscale=4 )
leg_lines=leg.get_lines()
plt.setp(leg_lines, linewidth=10)
plt.savefig('countour_beam.png')        # save png picture
h5.close()                              #close h5 file
