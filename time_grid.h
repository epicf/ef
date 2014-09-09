#ifndef _TIME_GRID_H_
#define _TIME_GRID_H_

#include <cstdio>
#include <cmath>
#include "config.h"

class Time_grid {
  private:
    //Time_grid() {};
  public:
    double total_time, current_time;
    double time_step_size;
    double time_save_step;
    int total_nodes, current_node, node_to_save;
    Time_grid( Config *conf );
    Time_grid( double total_time, double time_step_size, double time_save_step );
    void print( );
    void write_to_file( FILE *f );
}; 

#endif /* _TIME_GRID_H_ */
