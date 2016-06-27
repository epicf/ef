#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <mpi.h>
#include <hdf5.h>
#include "config.h"
#include "time_grid.h"
#include "spatial_mesh.h"
#include "inner_region.h"
#include "particle_to_mesh_map.h"
#include "field_solver.h"
#include "External_magnetic_field.h"
#include "particle_interaction_model.h"
#include "particle_source.h"
#include "particle.h"
#include "vec3d.h"

//#define M_PI 3.14159265358979323846264338327

class Domain {
  private:
    //Domain() {};
  public:
    Time_grid time_grid;
    Spatial_mesh spat_mesh;
    Inner_regions_manager inner_regions;
    Particle_to_mesh_map particle_to_mesh_map;
    Field_solver field_solver;    
    Particle_sources_manager particle_sources;
    External_magnetic_field external_magnetic_field;
    Particle_interaction_model particle_interaction_model;
  public:
    Domain( Config &conf );
    void run_pic( Config &conf );
    void eval_and_write_fields_without_particles( Config &conf );
    void write_step_to_save( Config &conf );
    void write( Config &conf );
    virtual ~Domain();
  private:
    // Pic algorithm
    void prepare_leap_frog();
    void advance_one_time_step();
    void eval_charge_density();
    void eval_potential_and_fields();
    void push_particles();
    void apply_domain_constrains();
    void remove_particles_inside_inner_regions();
    void update_time_grid();
    // Push particles
    void leap_frog();
    void shift_velocities_half_time_step_back();
    void update_momentum( double dt );
    void update_position( double dt );
    // Boundaries and generation
    void apply_domain_boundary_conditions();
    bool out_of_bound( const Particle &p );
    void generate_new_particles();    
    // Various functions
    void print_particles();
    bool negative( hid_t hdf5_id );
    void hdf5_status_check( herr_t status );
};

#endif /* _DOMAIN_H_ */
