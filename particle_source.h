#ifndef _PARTICLE_SOURCE_H_
#define _PARTICLE_SOURCE_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <vector>
#include "config.h"
#include "particle.h"

class Particle_source {
public:
    std::vector<Particle> particles;
public:
    Particle_source( Config *conf );
    void test_init( Config *conf );
    void print_all();
    void write_to_file( FILE *f );
    virtual ~Particle_source() {};
private:
    // Check config
    void check_correctness_of_related_config_fields( Config *conf );
    void particle_source_number_of_particles_gt_zero( Config *conf );
    void particle_source_x_left_ge_zero( Config *conf );
    void particle_source_x_left_le_particle_source_x_right( Config *conf );
    void particle_source_x_right_le_grid_x_size( Config *conf );
    void particle_source_y_bottom_ge_zero( Config *conf );
    void particle_source_y_bottom_le_particle_source_y_top( Config *conf );
    void particle_source_y_top_le_grid_y_size( Config *conf );
    void particle_source_temperature_gt_zero( Config *conf );
    void particle_source_mass_gt_zero( Config *conf );
};

#endif /* _PARTICLE_SOURCE_H_ */
