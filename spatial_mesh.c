#include "spatial_mesh.h"

void init_x_grid( Spatial_mesh *spm, const double x_size, const double x_step );
void init_y_grid( Spatial_mesh *spm, const double y_size, const double y_step );
void allocate_arrays_for_ongrid_values( Spatial_mesh *spm );
void print_grid( const Spatial_mesh *spm );
void print_ongrid_values( const Spatial_mesh *spm );

Spatial_mesh spatial_mesh_init( const double x_size, const double x_step,
				const double y_size, const double y_step )
{
  Spatial_mesh spm;
  init_x_grid( &spm, x_size, x_step );
  init_y_grid( &spm, y_size, y_step );
  allocate_arrays_for_ongrid_values( &spm );
  return spm;
}

void init_x_grid( Spatial_mesh *spm, const double x_size, const double x_step )
{
    spm->x_volume_size = x_size;
    spm->x_n_nodes = ceil( x_size / x_step) + 1;
    spm->x_cell_size = spm->x_volume_size / ( spm->x_n_nodes - 1 );
    if ( spm->x_cell_size != x_step ) {
	printf( "X_step was shrinked to %.3f from %.3f to fit round number of cells \n",
		spm->x_cell_size, x_step );
    }    
    return;
}

void init_y_grid( Spatial_mesh *spm, const double y_size, const double y_step )
{
    spm->y_volume_size = y_size;
    spm->y_n_nodes = ceil( y_size / y_step) + 1;
    spm->y_cell_size = spm->y_volume_size / ( spm->y_n_nodes -1 );
    if ( spm->y_cell_size != y_step ) {
	printf( "Y_step was shrinked to %.3f from %.3f to fit round number of cells \n",
		spm->y_cell_size, y_step );
    }    
    return;
}

void allocate_arrays_for_ongrid_values( Spatial_mesh *spm )
{
    int nx = spm->x_n_nodes;
    int ny = spm->y_n_nodes;    
    spm->charge_density = (double **) malloc( nx * sizeof(double *) );
    spm->potential = (double **) malloc( nx * sizeof(double *) );
    spm->electric_field = (Vec2d **) malloc( nx * sizeof(Vec2d *) );
    if ( ( spm->charge_density == NULL ) || 
	 ( spm->potential == NULL ) || 
	 ( spm->electric_field == NULL ) ) {
	printf( "allocate_arrays_for_ongrid_values: rows: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	spm->charge_density[i] = (double *) calloc( ny, sizeof(double) );
	spm->potential[i] = (double *) calloc( ny, sizeof(double) );
	spm->electric_field[i] = (Vec2d *) calloc( ny, sizeof(Vec2d) );
	if ( ( spm->charge_density[i] == NULL ) || 
	     ( spm->potential[i] == NULL ) || 
	     ( spm->electric_field[i] == NULL ) ) {
	    printf( "allocate_arrays_for_ongrid_values: cols: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }
    return;
}

void spatial_mesh_print( const Spatial_mesh *spm )
{
    print_grid( spm );
    print_ongrid_values( spm );
    return;
}

void print_grid( const Spatial_mesh *spm )
{
    printf( "Grid:\n" );
    printf( "Length: x = %f, y = %f \n", spm->x_volume_size, spm->y_volume_size );
    printf( "Cell size: x = %f, y = %f \n", spm->x_cell_size, spm->y_cell_size );
    printf( "Total nodes: x = %d, y = %d \n", spm->x_n_nodes, spm->y_n_nodes );
    return;
}

void print_ongrid_values( const Spatial_mesh *spm )
{
    int nx = spm->x_n_nodes;
    int ny = spm->y_n_nodes;
    printf( "(row, col): \t charge_density \t potential \t electric_field(x,y) \n");
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    printf( "(%d,%d): \t %.3f \t %.3f \t (%.3f,%.3f) \n",
		    i, j, 
		    spm->charge_density[i][j], spm->potential[i][j], 
		    vec2d_x( spm->electric_field[i][j] ), vec2d_y( spm->electric_field[i][j] ) );
	}
    }
    return;
}


void spatial_mesh_write_to_file( const Spatial_mesh *spm, FILE *f )
{
    fprintf(f, "%s", "Hello.\n");
    return;
}
