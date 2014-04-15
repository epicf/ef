#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "vec2d.h"
#include "config.h"


typedef struct {
    int id;
    double charge;
    double mass;
    Vec2d position;
    Vec2d momentum;
} Particle;

void particles_test_init( Particle **ps, int *num_of_particles, Config *conf );
void particle_print( const Particle *p );
void particle_print_all( const Particle *p, int n );
void particles_write_to_file( const Particle *p, const int num, FILE *f );

#endif /* _PARTICLES_H_ */
