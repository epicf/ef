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
Si_conv_G_T = 0.0001

def get_B_field( h5file ):
    B_field = h5file["/External_magnetic_field"].attrs["external_magnetic_field_z"][0]
    return B_field * Si_conv_G_T

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
    half_width_of_cathode = (start_x-end_x) / 2
    center_of_beam = (start_x+end_x) / 2    
    return ( length_of_cathode * SI_conv_cm_to_m, 
            half_width_of_cathode * SI_conv_cm_to_m, 
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
                                    
def eta(charge,mass):
    eta = np.abs(charge / mass )
    return eta

def velocity(eta,voltage):
    velocity  = np.sqrt(2*eta*voltage)
    return velocity    
                            
def R_const(half_thick, x0_const, velocity, angle, B):
    R_const = half_thick * np.sqrt( (1 - x0_const/half_thick)**2 + (velocity * np.tan(angle) / (eta * B * half_thick))**2)
    return R_const
    
    
def lambda_const(eta, voltage ,B_field):
    lambda_const = 4 * np.pi / (np.sqrt(2*eta)) * np.sqrt(voltage) / B_field
    return lambda_const
    
def phi_const(x0_const, half_thick, velocity, angle, eta, B_field):
    phi_const = -1 * np.arctan((1 - x0_const / half_thick) *  (eta * B_field * half_thick) / (velocity * np.tan(angle)))
    return phi_const

def x0_const(eta, current_dens, voltage, B_field, B_field_cathode,  xk):
    eps0 = 8.85e-12    
    a0 = 1 / (2*2**0.5*eps0*eta**(3/2)) * current_dens / (B_field**2*voltage**0.5)
    x0_const = a0 + B_field_cathode / B_field * xk
    return x0_const

def contour( z_position , x0_const, R_const, lambda_const, phi_const):
    contour = x0_const + R_const * np.sin(2*np.pi/lambda_const*z_position+phi_const)   
    return contour    



filename = "contour_0001000.h5"
h5 = h5py.File( filename, mode="r")

phi_shift = 1.6 # to combine phase
B_field = get_B_field( h5 )
B_field_cathode = B_field 
current = get_source_current( h5 )
mass, charge, momentum_z = get_source_particle_parameters( h5 )
length_of_cathode, half_thick, center_of_beam = get_source_geometry( h5 )
start_z, end_z = get_zlim( h5 )
voltage = get_voltage( momentum_z, mass, charge )
current_dens = get_current_dens(current,length_of_cathode)
eta  = eta(charge,mass)
velocity = velocity(eta,voltage)
conv_deg_to_rad = np.pi/180
angle = 0 * conv_deg_to_rad
x0_const = x0_const(eta, current_dens, voltage, B_field, B_field_cathode,  half_thick)
R_const = R_const(half_thick, x0_const, velocity, angle, B_field)
lambda_const = lambda_const(eta, voltage ,B_field)
phi_const  = phi_const(x0_const, half_thick, velocity, angle, eta, B_field) + phi_shift

steps_z = 100
position_z = np.arange(start_z,end_z,(end_z-start_z)/steps_z)                  # points in z direction, from 0 to 0.01 m with step 0,00001 m 

contour = contour( position_z , x0_const, R_const, lambda_const, phi_const)                   # countour calculation, m

h5 = h5py.File( filename , mode="r") # read h5 file
plt.xlabel("Z position,m")      
plt.ylabel("X position, m")
#plt.ylim(0.0002,0.0003)
plt.plot(h5["/Particle_sources/cathode_emitter/position_z"][:]*SI_conv_cm_to_m, 
            (h5["/Particle_sources/cathode_emitter/position_x"][:]*SI_conv_cm_to_m - center_of_beam),
             '.',label="calculated_points") #plot particles
plt.plot(position_z,contour, 'o',ms=4,label="analytic_curve")   # plot countour in cm and move to left z of beam and top x of beam neat cathode
plt.legend(bbox_to_anchor=(2.45, 1), loc=1, borderaxespad=0.)
plt.savefig('countour_beam.png')        # save png picture
h5.close()                              #close h5 file
