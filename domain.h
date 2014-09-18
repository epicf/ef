#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string>
#include <limits.h>
#include <vector>
#include <algorithm>
#include <functional>
#include "config.h"
#include "time_grid.h"
#include "spatial_mesh.h"
#include "field_solver.h"
#include "particle_source.h"
#include "particle.h"
#include "vec2d.h"

//#define M_PI 3.14159265358979323846264338327

class Domain {
  private:
    //Domain() {};
  public:
    Time_grid time_grid;
    Spatial_mesh spat_mesh;
    Field_solver field_solver;
    Particle_source part_src;
  public:
    Domain( Config *conf );
    void run_pic( Config *conf );
    void write_step_to_save( Config *conf );
    void write( Config *conf );
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
    // Eval charge density on grid
    void clear_old_density_values();
    void weight_particles_charge_to_mesh();
    //
    Vec2d force_on_particle( Particle &p );
    // Boundaries and generation
    void apply_domain_boundary_conditions();
    bool out_of_bound( const Particle &p );
    //bool out_of_bound( Vec2d r );
    // Various functions
    void print_particles();

};

#endif /* _DOMAIN_H_ */
