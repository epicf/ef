#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "time_grid.h"
#include "spatial_mesh.h"
#include "particles.h"
#include "vec2d.h"

typedef struct {
    Time_grid time_grid;
    Spatial_mesh spat_mesh;
    int num_of_particles;
    Particle *particles;
} Domain;

void domain_prepare( Domain *dom );
void domain_run_pic( Domain *dom );
void domain_write( Domain *dom );
void domain_free( Domain *dom );

#endif /* _DOMAIN_H_ */
