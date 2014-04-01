#ifndef _TIME_GRID_H_
#define _TIME_GRID_H_

#include <stdio.h>
#include <math.h>

typedef struct {
  double total_time, current_time;
  double time_step_size;
  int total_nodes, current_node;
} Time_grid;

Time_grid time_grid_init( double total_time, double time_step_size );
void time_grid_print( const Time_grid tg );
void time_grid_write_to_file( const Time_grid *tg, FILE *f );

#endif /* _TIME_GRID_H_ */
