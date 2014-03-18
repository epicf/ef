#ifndef _PARTICLES_H_
#define _PARTICLES_H_

#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "vec2d.h"


typedef struct {
    double charge;
    double mass;
    Vec2d position;
    Vec2d momentum;
} Particle;

void particles_test_init( Particle **ps, int *num_of_particles );
void particle_print( const Particle p );

#endif /* _PARTICLES_H_ */
