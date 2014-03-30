#include "time_grid.h"

Time_grid time_grid_init( double total_time, double time_step_size )
{
  Time_grid tg;
  tg.total_time = total_time;
  tg.total_nodes = ceil( total_time / time_step_size ) + 1;
  tg.time_step_size = tg.total_time / ( tg.total_nodes - 1 );
  if ( tg.time_step_size != time_step_size ) {
    printf( "Time step was shrinked to %.3f from %.3f to fit round number of cells", \
	    tg.time_step_size, time_step_size );
  }
  tg.current_time = 0.0;
  tg.current_node = 0;
  return tg;
}

void time_grid_print( Time_grid tg )
{
    printf( "Time grid:\n" );
    printf( "Total time = %f \n", tg.total_time );
    printf( "Current time = %f \n", tg.current_time );
    printf( "Time step size = %f \n", tg.time_step_size );
    printf( "Total nodes = %d \n", tg.total_nodes );
    printf( "Current node = %d \n", tg.current_node );
    return;
}

void time_grid_write_to_file( const Time_grid *tg, FILE *f )
{
    fprintf(f, "%s", "Hello.\n");
    return;
}
