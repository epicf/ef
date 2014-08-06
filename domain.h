#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "config.h"
#include "time_grid.h"
#include "spatial_mesh.h"
#include "particles.h"
#include "vec2d.h"

#define M_PI 3.14159265358979323846264338327

typedef struct {
    Time_grid time_grid;
    Spatial_mesh spat_mesh;
    int num_of_particles;
    Particle *particles;
} Domain;

void domain_prepare( Domain *dom, Config *conf );
void domain_run_pic( Domain *dom, Config *conf );
void domain_write_step_to_save( Domain *dom, Config *conf );
void domain_write( Domain *dom, Config *conf );
void domain_free( Domain *dom );

#endif /* _DOMAIN_H_ */
