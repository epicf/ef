#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include "config.h"
#include "time_grid.h"
#include "domain_geometry.h"
#include "field_solver.h"
#include "particle_source.h"
#include "particle.h"
#include "vec2d.h"

class Domain {
  private:
    //Domain() {};
  public:
    Time_grid time_grid;
    Domain_geometry domain_geometry;
    Particle_sources particle_sources;
    Field_solver<2> field_solver;    
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
    bool out_of_bound( const Particle &p );
    void generate_new_particles();
    // Various functions
    void print_particles();
};

#endif /* _DOMAIN_H_ */
