#include "domain.h"

// Domain print
std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix );

//
// Domain initialization
//

Domain::Domain( Config &conf ) :
    time_grid( conf ),
    domain_geometry( conf ),
    particle_sources( conf ),
    field_solver( conf, domain_geometry, particle_sources )
{    
    return;
}

//
// Pic simulation 
//

void Domain::run_pic( Config &conf )
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
    //eval_charge_density();
    eval_potential_and_fields();
    shift_velocities_half_time_step_back();
    return;
}

void Domain::advance_one_time_step()
{
    push_particles();
    apply_domain_constrains();
    //eval_charge_density();
    eval_potential_and_fields();
    update_time_grid();
    return;
}

// void Domain::eval_charge_density()
// {
//     spat_mesh.clear_old_density_values();
//     particle_to_mesh_map.weight_particles_charge_to_mesh( spat_mesh, particle_sources );
//     return;
// }

void Domain::eval_potential_and_fields()
{
    field_solver.eval_potential_and_fields();
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

    for( auto &src : particle_sources.sources ) {
	for( auto &p : src.particles ) {
	    force = field_solver.force_on_particle( p );
	    dp = vec2d_times_scalar( force, dt );
	    p.momentum = vec2d_add( p.momentum, dp );
	}
    }
    return;
}

void Domain::update_position( double dt )
{
    particle_sources.update_particles_position( dt );
    return;
}

//
// Apply domain constrains
//

void Domain::apply_domain_boundary_conditions()
{
    for( auto &src : particle_sources.sources ) {
	src.particles.erase( 
	    std::remove_if( 
		std::begin( src.particles ), 
		std::end( src.particles ), 
		[this]( Particle &p ){ return out_of_bound(p); } ), 
	    std::end( src.particles ) );
    }
    return;
}

bool Domain::out_of_bound( const Particle &p )
{
    double x = vec2d_x( p.position );
    double y = vec2d_y( p.position );
    bool out;
    
    out = 
	( x >= domain_geometry.x_volume_size ) || ( x <= 0 ) ||
	( y >= domain_geometry.y_volume_size ) || ( y <= 0 ) ;
    return out;

}

void Domain::generate_new_particles()
{
    particle_sources.generate_each_step();
    return;
}


//
// Update time grid
//

void Domain::update_time_grid()
{
    time_grid.update_to_next_step();
    return;
}

//
// Write domain to file
//

void Domain::write_step_to_save( Config &conf )
{
    int current_step = time_grid.current_node;
    int step_to_save = time_grid.node_to_save;
    if ( ( current_step % step_to_save ) == 0 ){	
	write( conf );
    }
    return;
}

void Domain::write( Config &conf )
{
    std::string output_filename_prefix = 
	conf.output_filename_config_part.output_filename_prefix;
    std::string output_filename_suffix = 
	conf.output_filename_config_part.output_filename_suffix;
    std::string file_name_to_write;
    
    file_name_to_write = construct_output_filename( output_filename_prefix, 
						    time_grid.current_node,
						    output_filename_suffix  );
			           
    std::ofstream output_file( file_name_to_write );
    if ( !output_file.is_open() ) {
	std::cout << "Error: can't open file \'" 
		  << file_name_to_write 
		  << "\' to save results of simulation!" 
		  << std::endl;
	std::cout << "Recheck \'output_filename_prefix\' key in config file." 
		  << std::endl;
	std::cout << "Make sure the directory you want to save to exists." 
		  << std::endl;
	exit( EXIT_FAILURE );
    }
    std::cout << "Writing step " << time_grid.current_node 
	      << " to file " << file_name_to_write << std::endl;
	    
    time_grid.write_to_file( output_file );
    domain_geometry.write_to_file( output_file );
    particle_sources.write_to_file( output_file );

    output_file.close();
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
    particle_sources.print_particles();
    return;
}
