#include "time_grid.h"

Time_grid::Time_grid( Config &conf ) 
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );
    init_total_nodes();
    shrink_time_step_size_if_necessary( conf ); 
    shrink_time_save_step_if_necessary( conf ); 
    set_current_time_and_node();
}

Time_grid::Time_grid( hid_t h5_time_grid_group )
{    
    herr_t status;
    status = H5LTget_attribute_double( h5_time_grid_group, "./",
				    "total_time", &total_time ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_time_grid_group, "./",
				       "current_time", &current_time ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_time_grid_group, "./",
				       "time_step_size", &time_step_size ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_time_grid_group, "./",
				       "time_save_step", &time_save_step ); hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_time_grid_group, "./",
				    "total_nodes", &total_nodes ); hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_time_grid_group, "./",
				    "current_node", &current_node ); hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_time_grid_group, "./",
				    "node_to_save", &node_to_save ); hdf5_status_check( status );

    status = H5Gclose(h5_time_grid_group); hdf5_status_check( status );
}

void Time_grid::check_correctness_of_related_config_fields( Config &conf )
{
    total_time_gt_zero( conf );
    time_step_size_gt_zero_le_total_time( conf );
    time_save_step_ge_time_step_size( conf );
}

void Time_grid::get_values_from_config( Config &conf )
{
    total_time = conf.time_config_part.total_time;
    time_step_size = conf.time_config_part.time_step_size; 
    time_save_step = conf.time_config_part.time_save_step;
}

void Time_grid::init_total_nodes()
{
    total_nodes = ceil( total_time / time_step_size ) + 1; 
}

void Time_grid::shrink_time_step_size_if_necessary( Config &conf )
{
    time_step_size = total_time / ( total_nodes - 1 );
    if ( time_step_size != conf.time_config_part.time_step_size ) {
	std::cout.precision(3);
	std::cout << "Time step was shrinked to " << time_step_size 
		  << " from " << conf.time_config_part.time_step_size 
		  << " to fit round number of cells." 
		  << std::endl;
    }
}

void Time_grid::shrink_time_save_step_if_necessary( Config &conf )
{
    time_save_step = ( (int)( time_save_step / time_step_size ) ) * time_step_size; 
    if ( time_save_step != conf.time_config_part.time_save_step ) {      
	std::cout.precision(3);
	std::cout << "Time save step was shrinked to " << time_save_step 
		  << " from " << conf.time_config_part.time_save_step 
		  << " to be a multiple of time step."
		  << std::endl;
    }
    node_to_save = (int) ( time_save_step / time_step_size );
}

void Time_grid::set_current_time_and_node()
{
    current_time = 0.0;
    current_node = 0;
}

void Time_grid::update_to_next_step()
{
    current_node++;
    current_time += time_step_size;
}

void Time_grid::print( )
{
    std::cout << "### Time grid:" << std::endl;
    std::cout << "Total time = " << total_time << std::endl;
    std::cout << "Current time = " << current_time << std::endl;
    std::cout << "Time step size = " << time_step_size << std::endl;
    std::cout << "Time save step = " << time_save_step << std::endl;
    std::cout << "Total nodes = " << total_nodes << std::endl;
    std::cout << "Current node = " << current_node << std::endl;
    std::cout << "Node to save = " << node_to_save << std::endl;
    return;
}

void Time_grid::write_to_file( hid_t hdf5_file_id )
{
    hid_t group_id;
    herr_t status;
    int single_element = 1;
    std::string hdf5_groupname = "/Time_grid";
    group_id = H5Gcreate( hdf5_file_id, hdf5_groupname.c_str(),
			  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); hdf5_status_check( group_id );

    status = H5LTset_attribute_double( hdf5_file_id, hdf5_groupname.c_str(),
				       "total_time", &total_time, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_double( hdf5_file_id, hdf5_groupname.c_str(),
				       "current_time", &current_time, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_double( hdf5_file_id, hdf5_groupname.c_str(),
				       "time_step_size", &time_step_size, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_double( hdf5_file_id, hdf5_groupname.c_str(),
				       "time_save_step", &time_save_step, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_int( hdf5_file_id, hdf5_groupname.c_str(),
				    "total_nodes", &total_nodes, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_int( hdf5_file_id, hdf5_groupname.c_str(),
				    "current_node", &current_node, single_element ); hdf5_status_check( status );
    status = H5LTset_attribute_int( hdf5_file_id, hdf5_groupname.c_str(),
				    "node_to_save", &node_to_save, single_element ); hdf5_status_check( status );
	
    status = H5Gclose(group_id); hdf5_status_check( status );
    return;
}

void Time_grid::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while writing Time_grid group. Aborting."
		  << std::endl;
	exit( EXIT_FAILURE );
    }
}

void Time_grid::total_time_gt_zero( Config &conf )
{
    check_and_exit_if_not( 
	conf.time_config_part.total_time >= 0, 
	"total_time < 0" );
}

void Time_grid::time_step_size_gt_zero_le_total_time( Config &conf )
{
    check_and_exit_if_not( 
	( conf.time_config_part.time_step_size > 0 ) && 
	( conf.time_config_part.time_step_size <= conf.time_config_part.total_time ),
	"time_step_size <= 0 or time_step_size > total_time" );
    return;
}

void Time_grid::time_save_step_ge_time_step_size( Config &conf )
{
    check_and_exit_if_not( 
	conf.time_config_part.time_save_step >= conf.time_config_part.time_step_size,
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
