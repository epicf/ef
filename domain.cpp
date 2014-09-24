#include "domain.h"

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
    particle_to_mesh_map( Particle_to_mesh_map() ),
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
    spat_mesh.clear_old_density_values();
    particle_to_mesh_map.weight_particles_charge_to_mesh( spat_mesh, part_src );
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

void Domain::apply_domain_constrains()
{
    apply_domain_boundary_conditions();
    generate_new_particles();
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
    Vec2d force, dp;

    for ( auto &p : part_src.particles ) {
	force = particle_to_mesh_map.force_on_particle( spat_mesh, p );
	dp = vec2d_times_scalar( force, dt );
	p.momentum = vec2d_add( p.momentum, dp );
    }
    return;
}

void Domain::update_position( double dt )
{
    Vec2d pos_shift;

    for ( auto &p : part_src.particles ) {
	pos_shift = vec2d_times_scalar( p.momentum, dt / p.mass );
	p.position = vec2d_add( p.position, pos_shift );
    }
    return;
}

//
// Apply domain constrains
//

void Domain::apply_domain_boundary_conditions()
{
    part_src.particles.erase( 
	std::remove_if( 
	    std::begin( part_src.particles ), 
	    std::end( part_src.particles ), 
	    [this]( Particle &p ){ return out_of_bound(p); } ), 
	std::end( part_src.particles ) );
  
    return;
}

bool Domain::out_of_bound( const Particle &p )
{
    double x = vec2d_x( p.position );
    double y = vec2d_y( p.position );
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= spat_mesh.y_volume_size ) || ( y <= 0 ) ;
    return out;

}

void Domain::generate_new_particles()
{
    part_src.generate_each_step();
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
    std::string output_filename_prefix = conf->output_filename_config_part.output_filename_prefix;
    std::string output_filename_suffix = conf->output_filename_config_part.output_filename_suffix;
    std::string file_name_to_write;
    
    file_name_to_write = construct_output_filename( output_filename_prefix, 
						    time_grid.current_node,
						    output_filename_suffix  );
			           
    FILE *f = fopen(file_name_to_write.c_str(), "w");
    if (f == NULL) {
	std::cout << "Error: can't open file \'" << file_name_to_write << "\' to save results of simulation!" 
		  << std::endl;
	std::cout << "Recheck \'output_filename_prefix\' key in config file." << std::endl;
	std::cout << "Make sure the directory you want to save to exists." << std::endl;
	exit( EXIT_FAILURE );
    }
    std::cout << "Writing step " << time_grid.current_node 
	      << " to file " << file_name_to_write << std::endl;
	    
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
    std::stringstream step_string;
    step_string << std::setfill('0') << std::setw(5) <<  current_time_step;

    std::string filename;
    filename = output_filename_prefix + 
	step_string.str() + 
	output_filename_suffix;
    return filename;
}

//
// Free domain
//

Domain::~Domain()
{
    std::cout << "TODO: free domain.\n";
    return;
}

//
// Various functions
//

void Domain::print_particles() 
{
    for ( auto &p : part_src.particles ) {
    	printf( "%d: (%.2f,%.2f), (%.2f,%.2f) \n", 
		p.id, 
		vec2d_x( p.position ),
		vec2d_y( p.position ),
		vec2d_x( p.momentum ),
		vec2d_y( p.momentum ));
    }
    return;
}
