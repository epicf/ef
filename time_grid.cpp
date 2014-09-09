#include "time_grid.h"

Time_grid::Time_grid( Config *conf ) 
  : Time_grid( conf->total_time, conf->time_step_size, conf->time_save_step ) {  };


Time_grid::Time_grid( double total_time, double time_step_size, double time_save_step )
{    
    this->total_time = total_time;
    total_nodes = ceil( total_time / time_step_size ) + 1; 
    //
    this->time_step_size = total_time / ( total_nodes - 1 );
    if ( this->time_step_size != time_step_size ) {
	printf( "Time step was shrinked to %.3f from %.3f to fit round number of cells.\n", \
		this->time_step_size, time_step_size );
    }
    //
    this->time_save_step = ( (int)( time_save_step / this->time_step_size ) ) * this->time_step_size; 
    if ( this->time_save_step != time_save_step ) {      
	printf( "Time save step was shrinked to %.3f from %.3f to be a multiple of time step.\n", \
		this->time_save_step, time_save_step );      
    }    
    node_to_save = (int) ( this->time_save_step / this->time_step_size );
    //
    current_time = 0.0;
    current_node = 0;
}

void Time_grid::print( )
{
    printf( "Time grid:\n" );
    printf( "Total time = %f \n", this->total_time );
    printf( "Current time = %f \n", this->current_time );
    printf( "Time step size = %f \n", this->time_step_size );
    printf( "Time save step = %f \n", this->time_save_step );    
    printf( "Total nodes = %d \n", this->total_nodes );
    printf( "Current node = %d \n", this->current_node );
    printf( "Node to save = %d \n", this->node_to_save );
    return;
}

void Time_grid::write_to_file( FILE *f )
{
    fprintf(f, "### Time grid\n" );
    fprintf(f, "Total time = %f \n", this->total_time );
    fprintf(f, "Current time = %f \n", this->current_time );
    fprintf(f, "Time step size = %f \n", this->time_step_size );
    fprintf(f, "Time save step = %f \n", this->time_save_step );
    fprintf(f, "Total nodes = %d \n", this->total_nodes );
    fprintf(f, "Current node = %d \n", this->current_node );
    fprintf(f, "Node to save = %d \n", this->node_to_save );
    return;
}
