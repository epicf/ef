#include "time_grid.h"

Time_grid time_grid_init( double total_time, double time_step_size, double time_save_step )
{
    Time_grid tg;
    tg.total_time = total_time;
    tg.total_nodes = ceil( total_time / time_step_size ) + 1;
    //
    tg.time_step_size = tg.total_time / ( tg.total_nodes - 1 );
    if ( tg.time_step_size != time_step_size ) {
	printf( "Time step was shrinked to %.3f from %.3f to fit round number of cells.\n", \
		tg.time_step_size, time_step_size );
    }
    //
    tg.time_save_step = ( (int)( time_save_step / tg.time_step_size ) ) * tg.time_step_size; 
    if ( tg.time_save_step != time_save_step ) {      
	printf( "Time save step was shrinked to %.3f from %.3f to be a multiple of time step.\n", \
		tg.time_save_step, time_save_step );      
    }    
    tg.node_to_save = (int) ( tg.time_save_step / tg.time_step_size );
    //
    tg.current_time = 0.0;
    tg.current_node = 0;
    return tg;
}

void time_grid_print( const Time_grid tg )
{
    printf( "Time grid:\n" );
    printf( "Total time = %f \n", tg.total_time );
    printf( "Current time = %f \n", tg.current_time );
    printf( "Time step size = %f \n", tg.time_step_size );
    printf( "Time save step = %f \n", tg.time_save_step );    
    printf( "Total nodes = %d \n", tg.total_nodes );
    printf( "Current node = %d \n", tg.current_node );
    printf( "Node to save = %d \n", tg.node_to_save );
    return;
}

void time_grid_write_to_file( const Time_grid *tg, FILE *f )
{
    fprintf(f, "### Time grid\n" );
    fprintf(f, "Total time = %f \n", tg->total_time );
    fprintf(f, "Current time = %f \n", tg->current_time );
    fprintf(f, "Time step size = %f \n", tg->time_step_size );
    fprintf(f, "Time save step = %f \n", tg->time_save_step );
    fprintf(f, "Total nodes = %d \n", tg->total_nodes );
    fprintf(f, "Current node = %d \n", tg->current_node );
    fprintf(f, "Node to save = %d \n", tg->node_to_save );
    return;
}
