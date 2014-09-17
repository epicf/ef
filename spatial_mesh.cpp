#include "spatial_mesh.h"

Spatial_mesh::Spatial_mesh( Config *conf )
{
    check_correctness_of_related_config_fields( conf );
    init_x_grid( conf );
    init_y_grid( conf );
    allocate_ongrid_values();
    set_boundary_conditions( conf );
}


void Spatial_mesh::check_correctness_of_related_config_fields( Config *conf )
{
    grid_x_size_gt_zero( conf );
    grid_x_step_gt_zero_le_grid_x_size( conf );
    grid_y_size_gt_zero( conf );
    grid_y_step_gt_zero_le_grid_y_size( conf );
}

void Spatial_mesh::init_x_grid( Config *conf )
{
    x_volume_size = conf->grid_x_size;
    x_n_nodes = ceil( conf->grid_x_size / conf->grid_x_step ) + 1;
    x_cell_size = x_volume_size / ( x_n_nodes - 1 );
    if ( x_cell_size != conf->grid_x_step ) {
	std::cout << "X_step was shrinked to " << x_cell_size << " from " 
		  << conf->grid_x_step << " to fit round number of cells" << std::endl;
    }    
    return;
}

void Spatial_mesh::init_y_grid( Config *conf )
{
    y_volume_size = conf->grid_y_size;
    y_n_nodes = ceil( conf->grid_y_size / conf->grid_y_step) + 1;
    y_cell_size = y_volume_size / ( y_n_nodes -1 );
    if ( y_cell_size != conf->grid_y_step ) {
	std::cout << "Y_step was shrinked to " << y_cell_size << " from " 
		  << conf->grid_y_step << " to fit round number of cells." << std::endl;
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
	printf( "allocate_arrays_for_ongrid_values: rows: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	charge_density[i] = (double *) calloc( ny, sizeof(double) );
	potential[i] = (double *) calloc( ny, sizeof(double) );
	electric_field[i] = (Vec2d *) calloc( ny, sizeof(Vec2d) );
	if ( ( charge_density[i] == NULL ) || 
	     ( potential[i] == NULL ) || 
	     ( electric_field[i] == NULL ) ) {
	    printf( "allocate_arrays_for_ongrid_values: cols: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }
    return;
}

void Spatial_mesh::set_boundary_conditions( Config *conf )
{
    set_boundary_conditions( conf->boundary_phi_left, conf->boundary_phi_right,
			     conf->boundary_phi_top, conf->boundary_phi_bottom );
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
    printf( "Grid:\n" );
    printf( "Length: x = %f, y = %f \n", x_volume_size, y_volume_size );
    printf( "Cell size: x = %f, y = %f \n", x_cell_size, y_cell_size );
    printf( "Total nodes: x = %d, y = %d \n", x_n_nodes, y_n_nodes );
    return;
}

void Spatial_mesh::print_ongrid_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    printf( "(row, col): \t charge_density \t potential \t electric_field(x,y) \n");
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    printf( "(%d,%d): \t %.3f \t %.3f \t (%.3f,%.3f) \n",
		    i, j, 
		    charge_density[i][j], potential[i][j], 
		    vec2d_x( electric_field[i][j] ), vec2d_y( electric_field[i][j] ) );
	}
    }
    return;
}

void Spatial_mesh::write_to_file( FILE *f )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;

    fprintf( f, "### Grid:\n" );
    fprintf( f, "X volume size = %f \n", x_volume_size );
    fprintf( f, "Y volume size = %f \n", y_volume_size );
    fprintf( f, "X cell size = %f \n", x_cell_size );
    fprintf( f, "Y cell size = %f \n", y_cell_size );
    fprintf( f, "X nodes = %d \n", x_n_nodes );
    fprintf( f, "Y nodes = %d \n", y_n_nodes );
    fprintf( f, "x_node  y_node  charge_density   potential \t electric_field(x,y) \n");
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    fprintf( f, "%-8d %-8d %-14.3f %-14.3f %-14.3f %-14.3f \n",
		    i, j, 
		    charge_density[i][j], 
		    potential[i][j], 
		    vec2d_x( electric_field[i][j] ), vec2d_y( electric_field[i][j] ) );
	}
    }
    return;
}

void Spatial_mesh::grid_x_size_gt_zero( Config *conf )
{
    check_and_exit_if_not( conf->grid_x_size > 0,
			   "grid_x_size < 0" );    
}

void Spatial_mesh::grid_x_step_gt_zero_le_grid_x_size( Config *conf )
{
    check_and_exit_if_not( conf->grid_x_step > 0 && conf->grid_x_step <= conf->grid_x_size,
			   "grid_x_step < 0 or grid_x_step >= grid_x_size" );    
}

void Spatial_mesh::grid_y_size_gt_zero( Config *conf )
{
    check_and_exit_if_not( conf->grid_y_size > 0,
			   "grid_y_size < 0" );    
}

void Spatial_mesh::grid_y_step_gt_zero_le_grid_y_size( Config *conf )
{
    check_and_exit_if_not( conf->grid_y_step > 0 && conf->grid_y_step <= conf->grid_y_size,
			   "grid_y_step < 0 or grid_y_step >= grid_y_size" );    
}


void Spatial_mesh::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " + message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}
