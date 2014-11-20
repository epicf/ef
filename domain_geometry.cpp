#include "domain_geometry.h"

Domain_geometry::Domain_geometry( Config &conf )    
{
    check_correctness_of_related_config_fields( conf );
    x_volume_size = conf.mesh_config_part.grid_x_size;
    y_volume_size = conf.mesh_config_part.grid_y_size;
    bottom_left = dealii::Point<2>( 0.0, 0.0 );
    top_right = dealii::Point<2>( x_volume_size, y_volume_size );
}

void Domain_geometry::check_correctness_of_related_config_fields( Config &conf )
{
    grid_x_size_gt_zero( conf );
    grid_y_size_gt_zero( conf );
}

void Domain_geometry::write_to_file( std::ofstream &output_file )
{    
    int x_n_nodes = 0;
    int y_n_nodes = 0;
    double x_cell_size = 0;
    double y_cell_size = 0;

    output_file << "### Grid" << std::endl;
    output_file << "X volume size = " << x_volume_size << std::endl;
    output_file << "Y volume size = " << y_volume_size << std::endl;
    output_file << "X cell size = " << x_cell_size << std::endl;
    output_file << "Y cell size = " << y_cell_size << std::endl;
    output_file << "X nodes = " << x_n_nodes << std::endl;
    output_file << "Y nodes = " << y_n_nodes << std::endl;
    output_file << "x_node \t\t y_node  charge_density   potential \t electric_field(x,y)" << std::endl;    
    output_file << "0 \t\t 0  0.0   0.0 \t 0.0 0.0" << std::endl;    
}

Domain_geometry::~Domain_geometry()
{
    // todo
}

bool Domain_geometry::at_left_boundary( const dealii::Point<2> &p ) const
{
    return std::fabs( p[0] - bottom_left[0] ) <= tolerance;
}

bool Domain_geometry::at_top_boundary( const dealii::Point<2> &p ) const
{
    return std::fabs( p[1] - top_right[1] ) <= tolerance;
}

bool Domain_geometry::at_right_boundary( const dealii::Point<2> &p ) const
{
    return std::fabs( p[0] - top_right[0] ) <= tolerance;
}

bool Domain_geometry::at_bottom_boundary( const dealii::Point<2> &p ) const
{
    return std::fabs( p[1] - bottom_left[1] ) <= tolerance;
}



// Config correctness

void Domain_geometry::grid_x_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_x_size > 0,
			   "grid_x_size < 0" );    
}

void Domain_geometry::grid_y_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_y_size > 0,
			   "grid_y_size < 0" );    
}

void Domain_geometry::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " << message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}
