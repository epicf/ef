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
void next_node_num_and_weight( const double x, int *next_node, double *weight );
// Eval fields from charges
void solve_poisson_eqn( Domain *dom );
void hwscrt_init_f( double left, double top, 
		    double right, double bottom, 
		    double **charge_density, double **hwscrt_f);
void check_pertrb();
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

void domain_prepare( Domain *dom )
{
    domain_time_grid_init( dom );    
    domain_spatial_mesh_init( dom );
    domain_particles_init( dom );
    return;
}

void domain_run_pic( Domain *dom )
{
    eval_charge_density( dom );
    /* eval_potential_and_fields( dom ); */
    /* push_particles( dom ); */
    /* apply_domain_constrains( dom ); */
    /* update_time_grid( dom ); */
    return;
}

void domain_write( Domain *dom )
{
    printf( "Hello." );
    return;
}

void domain_free( Domain *dom )
{
    printf( "TODO: free domain." );
    return;
}

//
// Domain initialization
//

void domain_time_grid_init( Domain *dom )
{
    double total_time = 1.0;
    double step_size = 0.1;
    dom->time_grid = time_grid_init( total_time, step_size );
    return;
}

void domain_spatial_mesh_init( Domain *dom )
{
    double x_size = 10.0;
    double x_step = 1;
    double y_size = 10.0;
    double y_step = 1;
    dom->spat_mesh = spatial_mesh_init( x_size, x_step, y_size, y_step );  
    return;
}

void domain_particles_init( Domain *dom )
{
    particles_test_init( &(dom->particles), &(dom->num_of_particles) );
    return;
}

//
// Pic algoritm
//

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
    //   charde[node] = weight(particle, node) * particle.charge
    // }
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;

    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( dom->particles[i].position ), 
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( dom->particles[i].position ), 
				  &tr_j, &tr_y_weight );
	dom->spat_mesh.charge_density[tr_i][tr_j] =			
	    tr_x_weight * tr_y_weight * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i-1][tr_j] =			
	    ( 1.0 - tr_x_weight ) * tr_y_weight * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i][tr_j-1] =			
	    tr_x_weight * ( 1.0 - tr_y_weight ) * dom->particles[i].charge;
	dom->spat_mesh.charge_density[tr_i-1][tr_j-1] =			
	    ( 1.0 - tr_x_weight ) * ( 1.0 - tr_y_weight )
	    * dom->particles[i].charge;      
    }
    return;
}

void next_node_num_and_weight( const double x, int *next_node, double *weight )
{
    *next_node = ceil( x );
    *weight = *next_node - x;
    return;
}

//
// Eval potential and fields
//

void solve_poisson_eqn( Domain *dom )
{
    /* double hwscrt_a = 0.0; */
    /* double hwscrt_b = dom->spat_mesh.x_volume_size; */
    /* int hwscrt_m = dom->spat_mesh.x_n_nodes-1; */
    /* int hwscrt_mbdcnd = 1; // 1st kind boundary conditions  */
    /* double hwscrt_bda = 0; // dummy */
    /* double hwscrt_bdb = 0; // dummy */
    /* double hwscrt_c = 0.0; */
    /* double hwscrt_d = dom->spat_mesh.y_volume_size; */
    /* int hwscrt_n = dom->spat_mesh.y_n_nodes-1; */
    /* int hwscrt_nbdcnd = 1; // 1st kind boundary conditions  */
    /* double hwscrt_bdc = 0; // dummy */
    /* double hwscrt_bdd = 0; // dummy */
    /* double hwscrt_elmdda = 0.0; */
    /* double hwscrt_f[hwscrt_m+1][hwscrt_n+1]; */
    /* //hwscrt_init_f(left, top, right, bottom, dom.spat_mesh.charge_density, hwscrt_f); */
    /* int hwscrt_idimf = hwscrt_m + 1; */
    /* int hwscrt_w_dim = 4 * ( hwscrt_n + 1 ) + ( 13 + (int)( log2( hwscrt_n + 1 ) ) ) * ( hwscrt_m + 1 ); */
    /* // double hwscrt_w[hwscrt_w_dim]; */
    /* // double hwscrt_pertrb; */
    /* int hwscrt_ierror; */
    /* // HWSCRT (A,B,M,MBDCND,BDA,BDB,C,D,N,NBDCND,BDC,BDD, ELMBDA,F,IDIMF,PERTRB,IERROR,W); */
    /* check_pertrb(); */
    /* if ( hwscrt_ierror != 0 ){ */
    /* 	printf( "Error while solving Poisson equation (HWSCRT): ierror = %d", hwscrt_ierror ); */
    /* 	exit( EXIT_FAILURE ); */
    /* } */
    /* // dom->spat_mesh.potential = hwscrt_f; */
    return;
}

void hwscrt_init_f( double left, double top, 
		    double right, double bottom, 
		    double **charge_density, double **hwscrt_f)
{
    printf( "TODO: hwscrt_init_fb" );
    return;
}


void check_pertrb()
{
    printf( "TODO: check_pertrb" );
    return;
}

void eval_fields_from_potential( Domain *dom )
{
    int nx = dom->spat_mesh.x_n_nodes;
    int ny = dom->spat_mesh.y_n_nodes;
    double dx = dom->spat_mesh.x_cell_size;
    double dy = dom->spat_mesh.y_cell_size;
    double ex[nx][ny], ey[nx][ny];

    for ( int j = 0; j < ny-1; j++ ) {
	for ( int i = 0; i < nx-1; i++ ) {
	    if ( i == 0 ) {
		ex[i][j] = - boundary_difference( ex[i][j], ex[i+1][j], dx );
	    } else if ( i == nx-1 ) {
		ex[i][j] = - boundary_difference( ex[i-1][j], ex[i][j], dx );
	    } else {
		ex[i][j] = - central_difference( ex[i-1][j], ex[i+1][j], dx );
	    }
	}
    }

    for ( int i = 0; i < nx-1; i++ ) {
	for ( int j = 0; j < ny-1; j++ ) {
	    if ( j == 0 ) {
		ey[i][j] = - boundary_difference( ex[i][j], ex[i][j+1], dy );
	    } else if ( j == ny-1 ) {
		ey[i][j] = - boundary_difference( ex[i][j-1], ex[i][j], dy );
	    } else {
		ey[i][j] = - central_difference( ex[i][j-1], ex[i][j+1], dy );
	    }
	}
    }

    for ( int i = 0; i < nx-1; i++ ) {
	for ( int j = 0; j < ny-1; j++ ) {
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
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;  
    Vec2d force, field_from_node;
    
    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( dom->particles[i].position ), 
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( dom->particles[i].position ), 
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
	force = vec2d_times_scalar( force, 1.0 );
	dom->particles[i].momentum = vec2d_add( dom->particles[i].momentum, force );
    }
    return;
}

void update_position( Domain *dom )
{
    Vec2d pos_shift;

    for ( int i = 0; i < dom->num_of_particles; i++ ) {
	pos_shift = vec2d_times_scalar( dom->particles[i].momentum, 1.0 );
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
    
    out = x > dom->spat_mesh.x_n_nodes || x < 0 ||
          y > dom->spat_mesh.y_n_nodes || y < 0;
    return out;
}

void remove_particle( int *i, Domain *dom )
{
    dom->particles[ *i ] = dom->particles[ *i - 1 ];
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
