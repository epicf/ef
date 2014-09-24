#include "time_grid.h"

Time_grid::Time_grid( Config *conf ) 
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );
    init_total_nodes();
    shrink_time_step_size_if_necessary( conf ); 
    shrink_time_save_step_if_necessary( conf ); 
    set_current_time_and_node();
}

void Time_grid::check_correctness_of_related_config_fields( Config *conf )
{
    total_time_gt_zero( conf );
    time_step_size_gt_zero_le_total_time( conf );
    time_save_step_ge_time_step_size( conf );
}

void Time_grid::get_values_from_config( Config *conf )
{
    total_time = conf->time_config_part.total_time;
    time_step_size = conf->time_config_part.time_step_size; 
    time_save_step = conf->time_config_part.time_save_step;
}

void Time_grid::init_total_nodes()
{
    total_nodes = ceil( total_time / time_step_size ) + 1; 
}

void Time_grid::shrink_time_step_size_if_necessary( Config *conf )
{
    time_step_size = total_time / ( total_nodes - 1 );
    if ( time_step_size != conf->time_config_part.time_step_size ) {
	std::cout << "Time step was shrinked to " << time_step_size 
		  << " from " << conf->time_config_part.time_step_size << " to fit round number of cells." 
		  << std::endl;
    }
}

void Time_grid::shrink_time_save_step_if_necessary( Config *conf )
{
    time_save_step = ( (int)( time_save_step / time_step_size ) ) * time_step_size; 
    if ( time_save_step != conf->time_config_part.time_save_step ) {      
	std::cout << "Time save step was shrinked to " << time_save_step 
		  << " from " << conf->time_config_part.time_save_step << " to be a multiple of time step."
		  << std::endl;
    }
    node_to_save = (int) ( time_save_step / time_step_size );
}

void Time_grid::set_current_time_and_node()
{
    current_time = 0.0;
    current_node = 0;
}

void Time_grid::print( )
{
    printf( "Time grid:\n" );
    printf( "Total time = %f \n", total_time );
    printf( "Current time = %f \n", current_time );
    printf( "Time step size = %f \n", time_step_size );
    printf( "Time save step = %f \n", time_save_step );    
    printf( "Total nodes = %d \n", total_nodes );
    printf( "Current node = %d \n", current_node );
    printf( "Node to save = %d \n", node_to_save );
    return;
}

void Time_grid::write_to_file( FILE *f )
{
    fprintf(f, "### Time grid\n" );
    fprintf(f, "Total time = %f \n", total_time );
    fprintf(f, "Current time = %f \n", current_time );
    fprintf(f, "Time step size = %f \n", time_step_size );
    fprintf(f, "Time save step = %f \n", time_save_step );
    fprintf(f, "Total nodes = %d \n", total_nodes );
    fprintf(f, "Current node = %d \n", current_node );
    fprintf(f, "Node to save = %d \n", node_to_save );
    return;
}

void Time_grid::total_time_gt_zero( Config *conf )
{
    check_and_exit_if_not( 
	conf->time_config_part.total_time >= 0, 
	"total_time < 0" );
}

void Time_grid::time_step_size_gt_zero_le_total_time( Config *conf )
{
    check_and_exit_if_not( 
	( conf->time_config_part.time_step_size > 0 ) && 
	( conf->time_config_part.time_step_size <= conf->time_config_part.total_time ),
	"time_step_size <= 0 or time_step_size > total_time" );
    return;
}

void Time_grid::time_save_step_ge_time_step_size( Config *conf )
{
    check_and_exit_if_not( 
	conf->time_config_part.time_save_step >= conf->time_config_part.time_step_size,
	"time_save_step < time_step_size" );
    return;
}

void Time_grid::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " + message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}
