#ifndef _TIME_GRID_H_
#define _TIME_GRID_H_

#include <cmath>
#include <iostream>
#include <string>
#include "Config.hpp"

template< int dim >
class Time_grid {
  public:
    double total_time, current_time;
    double time_step_size;
    double time_save_step;
    int total_nodes, current_node, node_to_save;
  public:
    Time_grid( Config<dim> &conf );
    void update_to_next_step();
    void print();
    void write_to_file( std::ofstream &output_file );
  private:
    // initialisation
    void check_correctness_of_related_config_fields( Config<dim> &conf );
    void get_values_from_config( Config<dim> &conf );
    void init_total_nodes();
    void shrink_time_step_size_if_necessary( Config<dim> &conf );
    void shrink_time_save_step_if_necessary( Config<dim> &conf );
    void set_current_time_and_node();
    // check config correctness
    void total_time_gt_zero( Config<dim> &conf );
    void time_step_size_gt_zero_le_total_time( Config<dim> &conf );
    void time_save_step_ge_time_step_size( Config<dim> &conf );
    void check_and_exit_if_not( const bool &should_be, const std::string &message );
}; 

template< int dim >
Time_grid<dim>::Time_grid( Config<dim> &conf ) 
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );
    init_total_nodes();
    shrink_time_step_size_if_necessary( conf ); 
    shrink_time_save_step_if_necessary( conf ); 
    set_current_time_and_node();
}

template< int dim >
void Time_grid<dim>::check_correctness_of_related_config_fields( Config<dim> &conf )
{
    total_time_gt_zero( conf );
    time_step_size_gt_zero_le_total_time( conf );
    time_save_step_ge_time_step_size( conf );
}

template< int dim >
void Time_grid<dim>::get_values_from_config( Config<dim> &conf )
{
    total_time = conf.time_config_part.total_time;
    time_step_size = conf.time_config_part.time_step_size; 
    time_save_step = conf.time_config_part.time_save_step;
}

template< int dim >
void Time_grid<dim>::init_total_nodes()
{
    total_nodes = ceil( total_time / time_step_size ) + 1; 
}

template< int dim >
void Time_grid<dim>::shrink_time_step_size_if_necessary( Config<dim> &conf )
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

template< int dim >
void Time_grid<dim>::shrink_time_save_step_if_necessary( Config<dim> &conf )
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

template< int dim >
void Time_grid<dim>::set_current_time_and_node()
{
    current_time = 0.0;
    current_node = 0;
}

template< int dim >
void Time_grid<dim>::update_to_next_step()
{
    current_node++;
    current_time += time_step_size;
}

template< int dim >
void Time_grid<dim>::print( )
{
    std::cout << "Time grid:" << std::endl;
    std::cout << "Total time = " << total_time << std::endl;
    std::cout << "Current time = " << current_time << std::endl;
    std::cout << "Time step size = " << time_step_size << std::endl;
    std::cout << "Time save step = " << time_save_step << std::endl;
    std::cout << "Total nodes = " << total_nodes << std::endl;
    std::cout << "Current node = " << current_node << std::endl;
    std::cout << "Node to save = " << node_to_save << std::endl;
    return;
}

template< int dim >
void Time_grid<dim>::write_to_file( std::ofstream &output_file )
{
    output_file << "Time grid:" << std::endl;
    output_file << "Total time = " << total_time << std::endl;
    output_file << "Current time = " << current_time << std::endl;
    output_file << "Time step size = " << time_step_size << std::endl;
    output_file << "Time save step = " << time_save_step << std::endl;
    output_file << "Total nodes = " << total_nodes << std::endl;
    output_file << "Current node = " << current_node << std::endl;
    output_file << "Node to save = " << node_to_save << std::endl;
    return;
}

template< int dim >
void Time_grid<dim>::total_time_gt_zero( Config<dim> &conf )
{
    check_and_exit_if_not( 
	conf.time_config_part.total_time >= 0, 
	"total_time < 0" );
}

template< int dim >
void Time_grid<dim>::time_step_size_gt_zero_le_total_time( Config<dim> &conf )
{
    check_and_exit_if_not( 
	( conf.time_config_part.time_step_size > 0 ) && 
	( conf.time_config_part.time_step_size <= conf.time_config_part.total_time ),
	"time_step_size <= 0 or time_step_size > total_time" );
    return;
}

template< int dim >
void Time_grid<dim>::time_save_step_ge_time_step_size( Config<dim> &conf )
{
    check_and_exit_if_not( 
	conf.time_config_part.time_save_step >= conf.time_config_part.time_step_size,
	"time_save_step < time_step_size" );
    return;
}

template< int dim >
void Time_grid<dim>::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " + message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}

#endif /* _TIME_GRID_H_ */
