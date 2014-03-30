#ifndef _SPATIAL_MESH_H_
#define _SPATIAL_MESH_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "vec2d.h"

typedef struct {
  double x_volume_size, y_volume_size;
  double x_cell_size, y_cell_size;
  int x_n_nodes, y_n_nodes;
  double **charge_density;
  double **potential;
  Vec2d **electric_field;
} Spatial_mesh;

Spatial_mesh spatial_mesh_init( const double x_size, const double x_step,
				const double y_size, const double y_step );
void spatial_mesh_print( const Spatial_mesh *spm );
void spatial_mesh_write_to_file( const Spatial_mesh *spm, FILE *f );

#endif /* _SPATIAL_MESH_H_ */
