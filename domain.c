#include "domain.h"

// Domain initialization
void domain_time_grid_init( Domain *dom );
void domain_spatial_mesh_init( Domain *dom );
void domain_particles_init( Domain *dom );
// Pic algorithm
void eval_charge_density( Domain *dom );
void eval_potential_and_fields( Domain *dom );
void push_particles( Domain *dom );
void apply_domain_constrains( Domain *dom );
void update_time_grid( Domain *dom );
// Eval charge density on grid
void weight_particles_charge_to_mesh( Domain *dom );
void next_node_num_and_weight( const double x, const double grid_step, int *next_node, double *weight );
// Eval fields from charges
void solve_poisson_eqn( Domain *dom );
extern void hwscrt_( double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, double *, int *, double * );
void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 );
void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 );		  
void hwscrt_init_f( double left, double top, 
		    double right, double bottom, 
		    double **charge_density, double **hwscrt_f);
double **poisson_init_rhs( double left, double top, 
			   double right, double bottom, 
			   double **charge_density, 
			   int nrow, int ncol );
void poisson_free_rhs( double **rhs, int nrow, int ncol );
void eval_fields_from_potential( Domain *dom );
double boundary_difference( double phi1, double phi2, double dx );
double central_difference( double phi1, double phi2, double dx );
// Push particles
void leap_frog( Domain *dom );
void update_momentum( Domain *dom );
void update_position( Domain *dom );
// Boundaries and generation
void apply_domain_boundary_conditions( Domain *dom );
bool out_of_bound( Domain *dom, Vec2d r );
void remove_particle( int *i, Domain *dom );
void proceed_to_next_particle( int *i, Domain *dom );
// Domain print
char *construct_output_filename( const char *output_filename_prefix, 
				 const int current_time_step,
				 const char *output_filename_suffix );
// Various functions
void domain_print_particles( Domain *dom );





//
// Domain initialization
//

void domain_prepare( Domain *dom )
{
    domain_time_grid_init( dom );    
    domain_spatial_mesh_init( dom );
    domain_particles_init( dom );
    return;
}

void domain_time_grid_init( Domain *dom )
{
    double total_time = 1.0;
    double step_size = 0.1;
    dom->time_grid = time_grid_init( total_time, step_size );
    return;
}

void domain_spatial_mesh_init( Domain *dom )
{
    double x_size = 30.0;
    double x_step = 1.0;
    double y_size = 20.0;
    double y_step = 1.0;
    dom->spat_mesh = spatial_mesh_init( x_size, x_step, y_size, y_step );  
    return;
}

void domain_particles_init( Domain *dom )
{
    particles_test_init( &(dom->particles), &(dom->num_of_particles) );
    return;
}

//
// Pic algorithm
//

void domain_run_pic( Domain *dom )
{
    eval_charge_density( dom );
    eval_potential_and_fields( dom );
    push_particles( dom );
    apply_domain_constrains( dom );
    update_time_grid( dom );
    return;
}

void eval_charge_density( Domain *dom )
{
    weight_particles_charge_to_mesh( dom );
    return;
}

void eval_potential_and_fields( Domain *dom )
{
    solve_poisson_eqn( dom );
    eval_fields_from_potential( dom );
    return;
}

void push_particles( Domain *dom )
{
    leap_frog( dom );
    return;
}

void apply_domain_constrains( Domain *dom )
{
    apply_domain_boundary_conditions( dom );
    //generate_new_particles();
    return;
}

//
// Eval charge density
//

void weight_particles_charge_to_mesh( Domain *dom )
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   charge[node] = weight(particle, node) * particle.charge
    // }
    double dx = dom->spat_mesh.x_cell_size;
    double dy = dom->spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;

    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( dom->particles[i].position ), dx, 
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( dom->particles[i].position ), dy,
				  &tr_j, &tr_y_weight );
	dom->spat_mesh.charge_density[tr_i][tr_j] +=
	    tr_x_weight * tr_y_weight * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i-1][tr_j] +=
	    ( 1.0 - tr_x_weight ) * tr_y_weight * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i][tr_j-1] +=
	    tr_x_weight * ( 1.0 - tr_y_weight ) * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i-1][tr_j-1] +=
	    ( 1.0 - tr_x_weight ) * ( 1.0 - tr_y_weight )
	    * dom->particles[i].charge;
    }
    return;
}

void next_node_num_and_weight( const double x, const double grid_step, 
			       int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = *next_node - x_in_grid_units;
    return;
}

//
// Eval potential and fields
//

void solve_poisson_eqn( Domain *dom )
{
    double a = 0.0;
    double b = dom->spat_mesh.x_volume_size;
    int nx = dom->spat_mesh.x_n_nodes;
    int M = nx-1;
    int MBDCND = 1; // 1st kind boundary conditions
    //
    double c = 0.0;
    double d = dom->spat_mesh.y_volume_size;
    int ny = dom->spat_mesh.y_n_nodes;
    int N = ny-1;
    int NBDCND = 1; // 1st kind boundary conditions
    //
    double BDA[ny]; // dummy
    double BDB[ny]; // dummy
    double BDC[nx]; // dummy
    double BDD[nx]; // dummy
    //
    double elmbda = 0.0;
    double **f_rhs = NULL;
    double hwscrt_f[ nx * ny ];
    int idimf = nx;
    //
    int w_dim = 
	4 * ( N + 1 ) + ( 13 + (int)( log2( N + 1 ) ) ) * ( M + 1 );
    double w[w_dim];
    double pertrb;
    int ierror;
    // Boundary potential values
    double left = 0.0;
    double top = 10.0;
    double right = 20.0;
    double bottom = 30.0; 

    f_rhs = poisson_init_rhs( left, top, right, bottom, 
			      dom->spat_mesh.charge_density,
			      nx, ny );
    rowmajor_to_colmajor( f_rhs, hwscrt_f, nx, ny );
    hwscrt_( 
	&a, &b, &M, &MBDCND, BDA, BDB,
	&c, &d, &N, &NBDCND, BDC, BDD,
	&elmbda, hwscrt_f, &idimf, &pertrb, &ierror, w);
    if ( ierror != 0 ){
	printf( "Error while solving Poisson equation (HWSCRT). \n" );
	printf( "ierror = %d \n", ierror );
    	exit( EXIT_FAILURE );
    }
    colmajor_to_rowmajor( hwscrt_f, dom->spat_mesh.potential, nx, ny );
    poisson_free_rhs( f_rhs, nx, ny );
    return;
}


double **poisson_init_rhs( double left, double top, 
			   double right, double bottom, 
			   double **charge_density, 
			   int nx, int ny )
{
    double **rhs = NULL;
    rhs = (double **) malloc( nx * sizeof(double *) );
    if ( rhs == NULL ) {
	printf( "f_rhs allocate: nx: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	rhs[i] = (double *) malloc( ny * sizeof(double) );
	if ( rhs[i] == NULL ) {
	    printf( "f_rhs allocate: ny: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }
    
    for ( int i = 1; i < nx-1; i++ ) {
	for ( int j = 1; j < ny-1; j++ ) {
	    rhs[i][j] = -4.0 * M_PI * charge_density[i][j];
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	rhs[i][0] = bottom;
	rhs[i][ny-1] = top;
    }

    for ( int j = 0; j < ny; j++ ) {
	rhs[0][j] = left;
	rhs[nx-1][j] = right;
    }
    
    return rhs;
}

void poisson_free_rhs( double **rhs, int nx, int ny )
{
    for( int i = 0; i < nx; i++) {
	free( rhs[i] );
    }
    free( rhs );
}

void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    *( fortran + i + ( j * dim1 ) ) = c[i][j];
	}
    }
    return;
}

void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    c[i][j] = *( fortran + i + ( j * dim1 ) );
	}
    }
    return;
}

void eval_fields_from_potential( Domain *dom )
{
    int nx = dom->spat_mesh.x_n_nodes;
    int ny = dom->spat_mesh.y_n_nodes;
    double dx = dom->spat_mesh.x_cell_size;
    double dy = dom->spat_mesh.y_cell_size;
    double **phi = dom->spat_mesh.potential;
    double ex[nx][ny], ey[nx][ny];

    for ( int j = 0; j < ny; j++ ) {
	for ( int i = 0; i < nx; i++ ) {
	    if ( i == 0 ) {
		ex[i][j] = - boundary_difference( phi[i][j], phi[i+1][j], dx );
	    } else if ( i == nx-1 ) {
		ex[i][j] = - boundary_difference( phi[i-1][j], phi[i][j], dx );
	    } else {
		ex[i][j] = - central_difference( phi[i-1][j], phi[i+1][j], dx );
	    }
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    if ( j == 0 ) {
		ey[i][j] = - boundary_difference( phi[i][j], phi[i][j+1], dy );
	    } else if ( j == ny-1 ) {
		ey[i][j] = - boundary_difference( phi[i][j-1], phi[i][j], dy );
	    } else {
		ey[i][j] = - central_difference( phi[i][j-1], phi[i][j+1], dy );
	    }
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    dom->spat_mesh.electric_field[i][j] = vec2d_init( ex[i][j], ey[i][j] );
	}
    }

    return;
}

double central_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

double boundary_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / dx );
}

//
// Push particles
//

void leap_frog( Domain *dom )
{  
    update_momentum( dom );
    update_position( dom );
    return;
}

void update_momentum( Domain *dom )
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   force[particle] = weight(particle, node) * force(node)
    // }
    double dt = dom->time_grid.time_step_size;
    double dx = dom->spat_mesh.x_cell_size;
    double dy = dom->spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;  
    Vec2d force, field_from_node;
    
    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( dom->particles[i].position ), dx,
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( dom->particles[i].position ), dy,
				  &tr_j, &tr_y_weight );
	//
	force = vec2d_zero();
	field_from_node = vec2d_times_scalar(			
	    dom->spat_mesh.electric_field[tr_i][tr_j],		
	    tr_x_weight );
	field_from_node = vec2d_times_scalar( field_from_node, tr_y_weight );
	force = vec2d_add( force, field_from_node );
	//
	field_from_node = vec2d_times_scalar(		        
	    dom->spat_mesh.electric_field[tr_i-1][tr_j],		
	    1.0 - tr_x_weight );
	field_from_node = vec2d_times_scalar( field_from_node, tr_y_weight );
	force = vec2d_add( force, field_from_node );
	//
	field_from_node = vec2d_times_scalar(			
	    dom->spat_mesh.electric_field[tr_i][tr_j - 1],	
	    tr_x_weight );
	field_from_node = vec2d_times_scalar( field_from_node, 1.0 - tr_y_weight );
	force = vec2d_add( force, field_from_node );
	//
	field_from_node = vec2d_times_scalar(			
	    dom->spat_mesh.electric_field[tr_i-1][tr_j-1],	
	    1.0 - tr_x_weight );
	field_from_node = vec2d_times_scalar( field_from_node, 1.0 - tr_y_weight );
	force = vec2d_add( force, field_from_node );
	//
	force = vec2d_times_scalar( force, dt * dom->particles[i].charge );
	dom->particles[i].momentum = vec2d_add( dom->particles[i].momentum, force );
    }
    return;
}

void update_position( Domain *dom )
{
    Vec2d pos_shift;
    double dt = dom->time_grid.time_step_size;

    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	pos_shift = vec2d_times_scalar( dom->particles[i].momentum, dt/dom->particles[i].mass );
	dom->particles[i].position = vec2d_add( dom->particles[i].position, pos_shift );
    }
    return;
}


//
// Apply domain constrains
//

void apply_domain_boundary_conditions( Domain *dom )
{
    int i = 0;
  
    while ( i < dom->num_of_particles ) {
	if ( out_of_bound( dom, dom->particles[i].position ) ) {
	    remove_particle( &i, dom );
	} else {
	    proceed_to_next_particle( &i, dom );
	}
    }  
    return;
}

bool out_of_bound( Domain *dom, Vec2d r )
{
    double x = vec2d_x( r );
    double y = vec2d_y( r );
    bool out;
    
    out = 
	( x >= dom->spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= dom->spat_mesh.y_volume_size ) || ( y <= 0 ) ;
    return out;
}

void remove_particle( int *i, Domain *dom )
{
    dom->particles[ *i ] = dom->particles[ dom->num_of_particles - 1 ];
    dom->num_of_particles--;
    return;
}

void proceed_to_next_particle( int *i, Domain *dom )
{
    (*i)++;	    
    return;
}

//
// Update time grid
//

void update_time_grid( Domain *dom )
{
    dom->time_grid.current_node++;
    dom->time_grid.current_time += dom->time_grid.time_step_size;
    return;
}


//
// Write domain to file
//

void domain_write( Domain *dom )
{
    const char output_filename_prefix[] = "out";
    const char output_filename_suffix[] = ".dat";
    char *file_name_to_write;
    
    file_name_to_write = construct_output_filename( output_filename_prefix, 
						    dom->time_grid.current_node,
						    output_filename_suffix  );
			           
    FILE *f = fopen(file_name_to_write, "w");
    if (f == NULL) {
	printf("Error opening file!\n");
	exit( EXIT_FAILURE );
    }
    printf ("Writing step %d to file %s\n", dom->time_grid.current_node, file_name_to_write);
	    
    time_grid_write_to_file( &(dom->time_grid), f );
    spatial_mesh_write_to_file( &(dom->spat_mesh), f );
    particles_write_to_file( dom->particles, dom->num_of_particles, f );

    free( file_name_to_write );
    fclose(f);
    return;
}

char *construct_output_filename( const char *output_filename_prefix, 
				 const int current_time_step,
				 const char *output_filename_suffix )
{    
    int prefix_len = strlen(output_filename_prefix);
    int suffix_len = strlen(output_filename_suffix);
    int number_len = ((CHAR_BIT * sizeof(int) - 1) / 3 + 2); // don't know how this works
    int ENOUGH = prefix_len + number_len + suffix_len;
    char *filename;
    filename = (char *) malloc( ENOUGH * sizeof(char) );
    snprintf(filename, ENOUGH, "%s%.4d%s", 
	     output_filename_prefix, current_time_step, output_filename_suffix);
    return filename;
}

void domain_free( Domain *dom )
{
    printf( "TODO: free domain.\n" );
    return;
}


//
// Various functions
//

void domain_print_particles( Domain *dom )
{
    for ( int i = 0; i < dom->num_of_particles; i++ ) {
    	printf( "%d: (%.2f,%.2f), (%.2f,%.2f) \n", 
		i, 
		vec2d_x(dom->particles[i].position),
		vec2d_y(dom->particles[i].position),
		vec2d_x(dom->particles[i].momentum),
		vec2d_y(dom->particles[i].momentum));
    }
    return;
}
