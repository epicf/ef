#include "spatial_mesh.h"















void Spatial_mesh::write_to_file( std::ofstream &output_file )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;

    output_file << "### Grid" << std::endl;
    output_file << "X volume size = " << x_volume_size << std::endl;
    output_file << "Y volume size = " << y_volume_size << std::endl;
    output_file << "X cell size = " << x_cell_size << std::endl;
    output_file << "Y cell size = " << y_cell_size << std::endl;
    output_file << "X nodes = " << x_n_nodes << std::endl;
    output_file << "Y nodes = " << y_n_nodes << std::endl;
    output_file << "x_node \t\t y_node  charge_density   potential \t electric_field(x,y)" << std::endl;
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 2 );
    output_file.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    output_file << std::setw(8) << std::left << i 
			<< std::setw(8) << std::left << j 
			<< std::setw(14) << charge_density[i][j]
			<< std::setw(14) << potential[i][j]
			<< std::setw(14) << electric_field[i][j].x()
			<< std::setw(14) << electric_field[i][j].y()
			<< std::endl;
	}
    }
    return;
}

void Spatial_mesh::grid_x_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_x_size > 0,
			   "grid_x_size < 0" );    
}

void Spatial_mesh::grid_x_step_gt_zero_le_grid_x_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_x_step > 0 ) && 
	( conf.mesh_config_part.grid_x_step <= conf.mesh_config_part.grid_x_size ),
			   "grid_x_step < 0 or grid_x_step >= grid_x_size" );    
}

void Spatial_mesh::grid_y_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_y_size > 0,
			   "grid_y_size < 0" );    
}

void Spatial_mesh::grid_y_step_gt_zero_le_grid_y_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_y_step > 0 ) && 
	( conf.mesh_config_part.grid_y_step <= conf.mesh_config_part.grid_y_size ),
			   "grid_y_step < 0 or grid_y_step >= grid_y_size" );    
}


void Spatial_mesh::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " << message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}

Spatial_mesh::~Spatial_mesh()
{
    int nx = x_n_nodes;
    for( int i = 0; i < nx; i++) {
	free( charge_density[i] );
	free( potential[i] );
	free( electric_field[i] );
    }
    free( charge_density );
    free( potential );
    free( electric_field );    
    return;
}
