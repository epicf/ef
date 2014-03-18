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
void time_grid_print( Time_grid tg );

#endif /* _TIME_GRID_H_ */
