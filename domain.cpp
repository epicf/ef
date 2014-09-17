#include "domain.h"

// Eval charge density on grid
void next_node_num_and_weight( const double x, const double grid_step, int *next_node, double *weight );
// Domain print
std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix );

//
// Domain initialization
//

Domain::Domain( Config *conf ) :
    time_grid( Time_grid( conf ) ),
    spat_mesh( Spatial_mesh( conf ) ),
    field_solver( Field_solver() ),
    part_src( Particle_source( conf ) )
{
    return;
}

//
// Pic simulation 
//

void Domain::run_pic( Config *conf )
{
    int total_time_iterations, current_node;
    total_time_iterations = time_grid.total_nodes - 1;
    current_node = time_grid.current_node;
    
    prepare_leap_frog();

    for ( int i = current_node; i < total_time_iterations; i++ ){
    	advance_one_time_step();
    	write_step_to_save( conf );
    }

    return;
}

void Domain::prepare_leap_frog()
{
    eval_charge_density();
    eval_potential_and_fields();
    shift_velocities_half_time_step_back();
    return;
}

void Domain::advance_one_time_step()
{
    push_particles();
    apply_domain_constrains();
    eval_charge_density();
    eval_potential_and_fields();
    update_time_grid();
    return;
}

void Domain::eval_charge_density()
{
    clear_old_density_values();
    weight_particles_charge_to_mesh();
    return;
}

void Domain::eval_potential_and_fields()
{
    field_solver.eval_potential( &spat_mesh );
    field_solver.eval_fields_from_potential( &spat_mesh );
    return;
}

void Domain::push_particles()
{
    leap_frog();
    return;
}

void Domain::apply_domain_constrains( )
{
    apply_domain_boundary_conditions( );
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

    for ( int i = 0; i < part_src.num_of_particles; i++ ) {
	next_node_num_and_weight( vec2d_x( part_src.particles[i].position ), dx, 
				  &tr_i, &tr_x_weight );
	next_node_num_and_weight( vec2d_y( part_src.particles[i].position ), dy,
				  &tr_j, &tr_y_weight );
	spat_mesh.charge_density[tr_i][tr_j] +=
	    tr_x_weight * tr_y_weight * part_src.particles[i].charge;
	spat_mesh.charge_density[tr_i-1][tr_j] +=
	    ( 1.0 - tr_x_weight ) * tr_y_weight * part_src.particles[i].charge;
	spat_mesh.charge_density[tr_i][tr_j-1] +=
	    tr_x_weight * ( 1.0 - tr_y_weight ) * part_src.particles[i].charge;
	spat_mesh.charge_density[tr_i-1][tr_j-1] +=
	    ( 1.0 - tr_x_weight ) * ( 1.0 - tr_y_weight )
	    * part_src.particles[i].charge;
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
// Push particles
//

void Domain::leap_frog()
{  
    double dt = time_grid.time_step_size;

    update_momentum( dt );
    update_position( dt );
    return;
}

void Domain::shift_velocities_half_time_step_back()
{
    double half_dt = time_grid.time_step_size / 2;

    update_momentum( -half_dt );
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

    for ( int i = 0; i < part_src.num_of_particles; i++ ) {
	force = force_on_particle( i );
	dp = vec2d_times_scalar( force, dt );
	part_src.particles[i].momentum = vec2d_add( part_src.particles[i].momentum, dp );
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
    next_node_num_and_weight( vec2d_x( part_src.particles[particle_number].position ), dx,
			      &tr_i, &tr_x_weight );
    next_node_num_and_weight( vec2d_y( part_src.particles[particle_number].position ), dy,
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
    force = vec2d_times_scalar( total_field, part_src.particles[particle_number].charge );
    return force;
}

void Domain::update_position( double dt )
{
    Vec2d pos_shift;

    for ( int i = 0; i < part_src.num_of_particles; i++ ) {
	pos_shift = vec2d_times_scalar( part_src.particles[i].momentum, dt/part_src.particles[i].mass );
	part_src.particles[i].position = vec2d_add( part_src.particles[i].position, pos_shift );
    }
    return;
}


//
// Apply domain constrains
//

void Domain::apply_domain_boundary_conditions()
{
    int i = 0;
  
    while ( i < part_src.num_of_particles ) {
	if ( out_of_bound( part_src.particles[i].position ) ) {
	    remove_particle( &i );
	} else {
	    proceed_to_next_particle( &i );
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
    part_src.particles[ *i ] = part_src.particles[ part_src.num_of_particles - 1 ];
    part_src.num_of_particles--;
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
	write( conf );
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
    part_src.write_to_file( f );

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
    for ( int i = 0; i < part_src.num_of_particles; i++ ) {
    	printf( "%d: (%.2f,%.2f), (%.2f,%.2f) \n", 
		i, 
		vec2d_x( part_src.particles[i].position ),
		vec2d_y( part_src.particles[i].position ),
		vec2d_x( part_src.particles[i].momentum ),
		vec2d_y( part_src.particles[i].momentum ));
    }
    return;
}
