#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include "Config.hpp"
#include "time_grid.h"
#include "Spatial_mesh.hpp"
#include "particle_to_mesh_map.h"
#include "field_solver.h"
#include "Particle_source.hpp"
#include "Particle.hpp"
#include "VecNd.hpp"

//#define M_PI 3.14159265358979323846264338327

template< int dim >
class Domain {
  private:
    //Domain() {};
  public:
    Time_grid time_grid;
    Spatial_mesh<dim> spat_mesh;
    Particle_to_mesh_map<dim> particle_to_mesh_map;
    Field_solver<dim> field_solver;    
    Particle_sources<dim> particle_sources;
  public:
    Domain( Config &conf );
    void run_pic( Config &conf );
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
    void update_time_grid();
    // Push particles
    void leap_frog();
    void shift_velocities_half_time_step_back();
    void update_momentum( double dt );
    void update_position( double dt );
    // Boundaries and generation
    void apply_domain_boundary_conditions();
    bool out_of_bound( const Particle<dim> &p );
    void generate_new_particles();
    // Various functions
    void print_particles();
};

#endif /* _DOMAIN_H_ */
