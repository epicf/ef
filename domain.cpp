#include "domain.h"

// Eval charge density on grid
void next_node_num_and_weight( const double x, const double grid_step, int *next_node, double *weight );
// Eval fields from charges
extern "C" void hwscrt_( double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, double *, int *, double * );
void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 );
void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 );		  
void hwscrt_init_f( double left, double top, 
		    double right, double bottom, 
		    double **charge_density, double **hwscrt_f);
double **poisson_init_rhs( Spatial_mesh *spm );
void poisson_free_rhs( double **rhs, int nrow, int ncol );
double boundary_difference( double phi1, double phi2, double dx );
double central_difference( double phi1, double phi2, double dx );
// Domain print
std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix );

//
// Domain initialization
//

Domain::Domain( Config *conf ) :
  time_grid( Time_grid( conf ) ),
  spat_mesh( Spatial_mesh( conf ) )
{
    config_check_correctness( conf );
    //
    this->particles_init( conf );
    return;
}

void Domain::particles_init( Config *conf )
{
    particles_test_init( &(this->particles), &(this->num_of_particles), conf );
    return;
}

//
// Pic simulation 
//

void Domain::run_pic( Config *conf )
{
    int total_time_iterations, current_node;
    total_time_iterations = this->time_grid.total_nodes - 1;
    current_node = this->time_grid.current_node;
    
    this->prepare_leap_frog();

    for ( int i = current_node; i < total_time_iterations; i++ ){
	this->advance_one_time_step();
	this->write_step_to_save( conf );
    }

    return;
}

void Domain::prepare_leap_frog()
{
    this->eval_charge_density();
    this->eval_potential_and_fields();
    this->shift_velocities_half_time_step_back();
    return;
}

void Domain::advance_one_time_step()
{
    this->push_particles();
    this->apply_domain_constrains();
    this->eval_charge_density();
    this->eval_potential_and_fields();
    this->update_time_grid();
    return;
}

void Domain::eval_charge_density()
{
    this->clear_old_density_values();
    this->weight_particles_charge_to_mesh();
    return;
}

void Domain::eval_potential_and_fields()
{
    this->solve_poisson_eqn();
    this->eval_fields_from_potential();
    return;
}

void Domain::push_particles()
{
    this->leap_frog();
    return;
}

void Domain::apply_domain_constrains( )
{
    this->apply_domain_boundary_conditions( );
    //generate_new_particles();
    return;
}

//
// Eval charge density
//

void Domain::clear_old_density_values()
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;    
    
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    spat_mesh.charge_density[i][j] = 0;
	}
    }
}


void Domain::weight_particles_charge_to_mesh()
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   charge[node] = weight(particle, node) * particle.charge
    // }
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;

    for ( int i = 0; i < num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( particles[i].position ), dx, 
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( particles[i].position ), dy,
				  &tr_j, &tr_y_weight );
	spat_mesh.charge_density[tr_i][tr_j] +=
	    tr_x_weight * tr_y_weight * particles[i].charge;
	spat_mesh.charge_density[tr_i-1][tr_j] +=
	    ( 1.0 - tr_x_weight ) * tr_y_weight * particles[i].charge;
	spat_mesh.charge_density[tr_i][tr_j-1] +=
	    tr_x_weight * ( 1.0 - tr_y_weight ) * particles[i].charge;
	spat_mesh.charge_density[tr_i-1][tr_j-1] +=
	    ( 1.0 - tr_x_weight ) * ( 1.0 - tr_y_weight )
	    * particles[i].charge;
    }
    return;
}

void next_node_num_and_weight( const double x, const double grid_step, 
			       int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}

//
// Eval potential and fields
//

void Domain::solve_poisson_eqn()
{
    double a = 0.0;
    double b = spat_mesh.x_volume_size;
    int nx = spat_mesh.x_n_nodes;
    int M = nx-1;
    int MBDCND = 1; // 1st kind boundary conditions
    //
    double c = 0.0;
    double d = spat_mesh.y_volume_size;
    int ny = spat_mesh.y_n_nodes;
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

    f_rhs = poisson_init_rhs( &( spat_mesh ) );
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
    colmajor_to_rowmajor( hwscrt_f, spat_mesh.potential, nx, ny );
    poisson_free_rhs( f_rhs, nx, ny );
    return;
}


double **poisson_init_rhs( Spatial_mesh *spm )
{
    int nx = spm->x_n_nodes;
    int ny = spm->y_n_nodes;    

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
	    rhs[i][j] = -4.0 * M_PI * spm->charge_density[i][j];
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	rhs[i][0] = spm->potential[i][0];
	rhs[i][ny-1] = spm->potential[i][ny-1];
    }

    for ( int j = 0; j < ny; j++ ) {
	rhs[0][j] = spm->potential[0][j];
	rhs[nx-1][j] = spm->potential[nx-1][j];
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

void Domain::eval_fields_from_potential()
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double **phi = spat_mesh.potential;
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
	    spat_mesh.electric_field[i][j] = vec2d_init( ex[i][j], ey[i][j] );
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

void Domain::leap_frog()
{  
    double dt = time_grid.time_step_size;

    this->update_momentum( dt );
    this->update_position( dt );
    return;
}

void Domain::shift_velocities_half_time_step_back()
{
    double half_dt = time_grid.time_step_size / 2;

    this->update_momentum( -half_dt );
    return;    
}

void Domain::update_momentum( double dt )
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   force[particle] = weight(particle, node) * force(node)
    // }
    Vec2d force, dp;

    for ( int i = 0; i < num_of_particles; i++ ) {
	force = this->force_on_particle( i );
	dp = vec2d_times_scalar( force, dt );
	particles[i].momentum = vec2d_add( particles[i].momentum, dp );
    }
    return;
}

Vec2d Domain::force_on_particle( int particle_number )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;  
    Vec2d field_from_node, total_field, force;
    //
    next_node_num_and_weight( vec2d_x( particles[particle_number].position ), dx,
			      &tr_i, &tr_x_weight );
    next_node_num_and_weight( vec2d_y( particles[particle_number].position ), dy,
			      &tr_j, &tr_y_weight );
    //
    total_field = vec2d_zero();
    field_from_node = vec2d_times_scalar(			
	spat_mesh.electric_field[tr_i][tr_j],		
	tr_x_weight );
    field_from_node = vec2d_times_scalar( field_from_node, tr_y_weight );
    total_field = vec2d_add( total_field, field_from_node );
    //
    field_from_node = vec2d_times_scalar(		        
	spat_mesh.electric_field[tr_i-1][tr_j],		
	1.0 - tr_x_weight );
    field_from_node = vec2d_times_scalar( field_from_node, tr_y_weight );
    total_field = vec2d_add( total_field, field_from_node );
    //
    field_from_node = vec2d_times_scalar(			
	spat_mesh.electric_field[tr_i][tr_j - 1],	
	tr_x_weight );
    field_from_node = vec2d_times_scalar( field_from_node, 1.0 - tr_y_weight );
    total_field = vec2d_add( total_field, field_from_node );
    //
    field_from_node = vec2d_times_scalar(			
	spat_mesh.electric_field[tr_i-1][tr_j-1],	
	1.0 - tr_x_weight );
    field_from_node = vec2d_times_scalar( field_from_node, 1.0 - tr_y_weight );
    total_field = vec2d_add( total_field, field_from_node );
    //
    force = vec2d_times_scalar( total_field, particles[particle_number].charge );
    return force;
}

void Domain::update_position( double dt )
{
    Vec2d pos_shift;

    for ( int i = 0; i < num_of_particles; i++ ) {
	pos_shift = vec2d_times_scalar( particles[i].momentum, dt/particles[i].mass );
	particles[i].position = vec2d_add( particles[i].position, pos_shift );
    }
    return;
}


//
// Apply domain constrains
//

void Domain::apply_domain_boundary_conditions()
{
    int i = 0;
  
    while ( i < num_of_particles ) {
	if ( out_of_bound( particles[i].position ) ) {
	    this->remove_particle( &i );
	} else {
	    this->proceed_to_next_particle( &i );
	}
    }  
    return;
}

bool Domain::out_of_bound( Vec2d r )
{
    double x = vec2d_x( r );
    double y = vec2d_y( r );
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= spat_mesh.y_volume_size ) || ( y <= 0 ) ;
    return out;
}

void Domain::remove_particle( int *i )
{
    particles[ *i ] = particles[ num_of_particles - 1 ];
    num_of_particles--;
    return;
}

void Domain::proceed_to_next_particle( int *i )
{
    (*i)++;	    
    return;
}

//
// Update time grid
//

void Domain::update_time_grid()
{
    time_grid.current_node++;
    time_grid.current_time += time_grid.time_step_size;
    return;
}


//
// Write domain to file
//

void Domain::write_step_to_save( Config *conf )
{
    int current_step = time_grid.current_node;
    int step_to_save = time_grid.node_to_save;
    if ( ( current_step % step_to_save ) == 0 ){	
	this->write( conf );
    }
    return;
}

void Domain::write( Config *conf )
{
    char *output_filename_prefix = conf->output_filename_prefix;
    char *output_filename_suffix = conf->output_filename_suffix;
    const char *file_name_to_write;
    
    file_name_to_write = construct_output_filename( output_filename_prefix, 
						    time_grid.current_node,
						    output_filename_suffix  ).c_str();
			           
    FILE *f = fopen(file_name_to_write, "w");
    if (f == NULL) {
	printf( "Error: can't open file \'%s\' to save results of simulation!\n", file_name_to_write );
	printf( "Recheck 'output_filename_prefix' key in config file.\n" );
	printf( "Make sure the directory you want to save to exists.\n" );
	exit( EXIT_FAILURE );
    }
    printf ("Writing step %d to file %s\n", time_grid.current_node, file_name_to_write);
	    
    time_grid.write_to_file( f );
    spat_mesh.write_to_file( f );
    particles_write_to_file( particles, num_of_particles, f );

    fclose(f);
    return;
}

std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix )
{    
    std::string filename;
    filename = output_filename_prefix + std::to_string( current_time_step ) + output_filename_suffix;
    return filename;
}

//
// Free domain
//

Domain::~Domain()
{
    printf( "TODO: free domain.\n" );
    return;
}


//
// Various functions
//

void Domain::print_particles() 
{
    for ( int i = 0; i < num_of_particles; i++ ) {
    	printf( "%d: (%.2f,%.2f), (%.2f,%.2f) \n", 
		i, 
		vec2d_x( particles[i].position ),
		vec2d_y( particles[i].position ),
		vec2d_x( particles[i].momentum ),
		vec2d_y( particles[i].momentum ));
    }
    return;
}
