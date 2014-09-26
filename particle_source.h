#ifndef _PARTICLE_SOURCE_H_
#define _PARTICLE_SOURCE_H_

#include <stdio.h>
#include <math.h>
#include <random>
#include <iostream>
#include <vector>
#include "config.h"
#include "particle.h"

class Single_particle_source{
private:
    int initial_number_of_particles;
    int particles_to_generate_each_step;
    // Source position
    double xleft;
    double xright;
    double ytop;
    double ybottom;
    // Momentum
    double temperature;
    // Particle characteristics
    double charge;
    double mass;
    // Random number generator
    std::default_random_engine rnd_gen;
public:
    std::vector<Particle> particles;
public:
    Single_particle_source( Config *conf, Source_config_part &src_conf  );
    void generate_each_step();
    void print_all();
    void write_to_file( FILE *f );
    virtual ~Single_particle_source() {};
private:
    // Particle initialization
    void set_parameters_from_config( Source_config_part &src_conf );
    void generate_initial_particles();
    // Todo: replace 'std::default_random_engine' type with something more general.
    void generate_num_of_particles( int num_of_particles );
    int generate_particle_id( const int number );
    Vec2d uniform_position_in_rectangle( const double xleft,  const double ytop,
					 const double xright, const double ybottom,
					 std::default_random_engine &rnd_gen );
    double random_in_range( const double low, const double up, std::default_random_engine &rnd_gen );
    Vec2d maxwell_momentum_distr( const double temperature, const double mass, 
				  std::default_random_engine &rnd_gen );
    // Check config
    void check_correctness_of_related_config_fields( Config *conf, Source_config_part &src_conf );
    void particle_source_initial_number_of_particles_gt_zero( Config *conf, Source_config_part &src_conf );
    void particle_source_particles_to_generate_each_step_ge_zero( Config *conf, Source_config_part &src_conf );
    void particle_source_x_left_ge_zero( Config *conf, Source_config_part &src_conf );
    void particle_source_x_left_le_particle_source_x_right( Config *conf, Source_config_part &src_conf );
    void particle_source_x_right_le_grid_x_size( Config *conf, Source_config_part &src_conf );
    void particle_source_y_bottom_ge_zero( Config *conf, Source_config_part &src_conf );
    void particle_source_y_bottom_le_particle_source_y_top( Config *conf, Source_config_part &src_conf );
    void particle_source_y_top_le_grid_y_size( Config *conf, Source_config_part &src_conf );
    void particle_source_temperature_gt_zero( Config *conf, Source_config_part &src_conf );
    void particle_source_mass_gt_zero( Config *conf, Source_config_part &src_conf );
};


class Particle_sources{
public:
    std::vector<Single_particle_source> sources;
public:
    Particle_sources( Config *conf );
    virtual ~Particle_sources() {};
};


#endif /* _PARTICLE_SOURCE_H_ */
