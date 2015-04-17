#include "spatial_mesh.h"

Spatial_mesh::Spatial_mesh( Config &conf )
{
    check_correctness_of_related_config_fields( conf );
    init_x_grid( conf );
    init_y_grid( conf );
    allocate_ongrid_values();
    set_boundary_conditions( conf );
}


void Spatial_mesh::check_correctness_of_related_config_fields( Config &conf )
{
    grid_x_size_gt_zero( conf );
    grid_x_step_gt_zero_le_grid_x_size( conf );
    grid_y_size_gt_zero( conf );
    grid_y_step_gt_zero_le_grid_y_size( conf );
}

void Spatial_mesh::init_x_grid( Config &conf )
{
    x_volume_size = conf.mesh_config_part.grid_x_size;
    x_n_nodes = 
	ceil( conf.mesh_config_part.grid_x_size / conf.mesh_config_part.grid_x_step ) + 1;
    x_cell_size = x_volume_size / ( x_n_nodes - 1 );
    if ( x_cell_size != conf.mesh_config_part.grid_x_step ) {
	std::cout.precision(3);
	std::cout << "X_step was shrinked to " << x_cell_size 
		  << " from " << conf.mesh_config_part.grid_x_step 
		  << " to fit round number of cells" << std::endl;
    }    
    return;
}

void Spatial_mesh::init_y_grid( Config &conf )
{
    y_volume_size = conf.mesh_config_part.grid_y_size;
    y_n_nodes = 
	ceil( conf.mesh_config_part.grid_y_size / conf.mesh_config_part.grid_y_step) + 1;
    y_cell_size = y_volume_size / ( y_n_nodes -1 );
    if ( y_cell_size != conf.mesh_config_part.grid_y_step ) {
	std::cout.precision(3);
	std::cout << "Y_step was shrinked to " << y_cell_size 
		  << " from " << conf.mesh_config_part.grid_y_step 
		  << " to fit round number of cells." << std::endl;
    }    
    return;
}

void Spatial_mesh::allocate_ongrid_values( )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;    
    charge_density = (double **) malloc( nx * sizeof(double *) );
    potential = (double **) malloc( nx * sizeof(double *) );
    electric_field = (Vec2d **) malloc( nx * sizeof(Vec2d *) );
    if ( ( charge_density == NULL ) || 
	 ( potential == NULL ) || 
	 ( electric_field == NULL ) ) {
	std::cout << "allocate_arrays_for_ongrid_values: rows: out of memory " 
		  << std::endl;
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	charge_density[i] = (double *) calloc( ny, sizeof(double) );
	potential[i] = (double *) calloc( ny, sizeof(double) );
	electric_field[i] = (Vec2d *) calloc( ny, sizeof(Vec2d) );
	if ( ( charge_density[i] == NULL ) || 
	     ( potential[i] == NULL ) || 
	     ( electric_field[i] == NULL ) ) {
	    std::cout << "allocate_arrays_for_ongrid_values: cols: out of memory " 
		      << std::endl;
	    exit( EXIT_FAILURE );	
	}
    }
    return;
}

void Spatial_mesh::clear_old_density_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;    
    
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    charge_density[i][j] = 0;
	}
    }
}


void Spatial_mesh::set_boundary_conditions( Config &conf )
{
    set_boundary_conditions( conf.boundary_config_part.boundary_phi_left, 
			     conf.boundary_config_part.boundary_phi_right,
			     conf.boundary_config_part.boundary_phi_top, 
			     conf.boundary_config_part.boundary_phi_bottom );
}


void Spatial_mesh::set_boundary_conditions( const double phi_left, const double phi_right,
					    const double phi_top, const double phi_bottom )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;    

    for ( int i = 0; i < nx; i++ ) {
	potential[i][0] = phi_bottom;
	potential[i][ny-1] = phi_top;
    }
    
    for ( int j = 0; j < ny; j++ ) {
	potential[0][j] = phi_left;
	potential[nx-1][j] = phi_right;
    }

    return;
}

void Spatial_mesh::print()
{
    print_grid();
    print_ongrid_values();
    return;
}

void Spatial_mesh::print_grid()
{
    std::cout << "Grid:" << std::endl;
    std::cout << "Length: x = " << x_volume_size << ", "
	      << "y = " << y_volume_size << std::endl;
    std::cout << "Cell size: x = " << x_cell_size << ", "
	      << "y = " << y_cell_size << std::endl;
    std::cout << "Total nodes: x = " << x_n_nodes << ", "
	      << "y = " << y_n_nodes << std::endl;
    return;
}

void Spatial_mesh::print_ongrid_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    std::cout << "x_node \t\t y_node  charge_density   potential \t electric_field(x,y)" << std::endl;
    std::cout.precision( 3 );
    std::cout.setf( std::ios::scientific );
    std::cout.fill(' ');
    std::cout.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    std::cout << std::setw(8) << i 
			<< std::setw(8) << j 
			<< std::setw(14) << charge_density[i][j]
			<< std::setw(14) << potential[i][j]
			<< std::setw(14) << vec2d_x( electric_field[i][j] ) 
			<< std::setw(14) << vec2d_y( electric_field[i][j] ) 
			<< std::endl;
	}
    }
    return;
}

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
			<< std::setw(14) << vec2d_x( electric_field[i][j] ) 
			<< std::setw(14) << vec2d_y( electric_field[i][j] ) 
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
