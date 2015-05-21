#ifndef _DOMAIN_H_
#define _DOMAIN_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include "Config.hpp"
#include "Time_grid.hpp"
#include "Spatial_mesh.hpp"
#include "Particle_to_mesh_map.hpp"
#include "Field_solver.hpp"
#include "Particle_source.hpp"
#include "Particle.hpp"
#include "VecNd.hpp"

//#define M_PI 3.14159265358979323846264338327

template< int dim >
class Domain {
  private:
    //Domain() {};
  public:
    Time_grid<dim> time_grid;
    Spatial_mesh<dim> spat_mesh;
    Particle_to_mesh_map<dim> particle_to_mesh_map;
    Field_solver<dim> field_solver;    
    Particle_sources<dim> particle_sources;
  public:
    Domain( Config<dim> &conf );
    void run_pic( Config<dim> &conf );
    void write_step_to_save( Config<dim> &conf );
    void write( Config<dim> &conf );
    virtual ~Domain();
  private:
    // Pic algorithm
    void prepare_leap_frog();
    void advance_one_time_step();
    void eval_charge_density();
    void eval_potential_and_fields();
    void push_particles();
    void apply_domain_constrains();
    void update_time_grid();
    // Push particles
    void leap_frog();
    void shift_velocities_half_time_step_back();
    void update_momentum( double dt );
    void update_position( double dt );
    // Boundaries and generation
    void apply_domain_boundary_conditions();
    bool out_of_bound( const Particle<dim> &p );
    void generate_new_particles();
    // Various functions
    void print_particles();
};

// Domain print
std::string construct_output_filename( const std::string output_filename_prefix, 
				       const int current_time_step,
				       const std::string output_filename_suffix );

//
// Domain initialization
//

template< int dim >
Domain<dim>::Domain( Config<dim> &conf ) :
    time_grid( conf ),
    spat_mesh( conf ),
    particle_to_mesh_map( ),
    field_solver( spat_mesh ),
    particle_sources( conf )
{
    return;
}

//
// Pic simulation 
//

template< int dim >
void Domain<dim>::run_pic( Config<dim> &conf )
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

template< int dim >
void Domain<dim>::prepare_leap_frog()
{
    eval_charge_density();
    eval_potential_and_fields();
    shift_velocities_half_time_step_back();
    return;
}

template< int dim >
void Domain<dim>::advance_one_time_step()
{
    push_particles();
    apply_domain_constrains();
    eval_charge_density();
    eval_potential_and_fields();
    update_time_grid();
    return;
}

template< int dim >
void Domain<dim>::eval_charge_density()
{
    spat_mesh.clear_old_density_values();
    particle_to_mesh_map.weight_particles_charge_to_mesh( spat_mesh, particle_sources );
    return;
}

template< int dim >
void Domain<dim>::eval_potential_and_fields()
{
    field_solver.eval_potential( spat_mesh );
    field_solver.eval_fields_from_potential( spat_mesh );
    return;
}

template< int dim >
void Domain<dim>::push_particles()
{
    leap_frog();
    return;
}

template< int dim >
void Domain<dim>::apply_domain_constrains()
{
    apply_domain_boundary_conditions();
    generate_new_particles();
    return;
}

//
// Push particles
//

template< int dim >
void Domain<dim>::leap_frog()
{  
    double dt = time_grid.time_step_size;

    update_momentum( dt );
    update_position( dt );
    return;
}

template< int dim >
void Domain<dim>::shift_velocities_half_time_step_back()
{
    double half_dt = time_grid.time_step_size / 2.0;

    update_momentum( -half_dt );
    return;    
}

template< int dim >
void Domain<dim>::update_momentum( double dt )
{
    VecNd<dim> force, dp;

    for( auto &src : particle_sources.sources ) {
	for( auto &p : src.particles ) {
	    force = particle_to_mesh_map.force_on_particle( spat_mesh, p );
	    dp = force * dt;
	    p.momentum = p.momentum + dp;
	}
    }
    return;
}

template< int dim >
void Domain<dim>::update_position( double dt )
{
    particle_sources.update_particles_position( dt );
    return;
}

//
// Apply domain constrains
//

template< int dim >
void Domain<dim>::apply_domain_boundary_conditions()
{
    for( auto &src : particle_sources.sources ) {
	src.particles.erase( 
	    std::remove_if( 
		std::begin( src.particles ), 
		std::end( src.particles ), 
		[this]( Particle<dim> &p ){ return out_of_bound(p); } ), 
	    std::end( src.particles ) );
    }
    return;
}

template<int dim>
bool Domain<dim>::out_of_bound( const Particle<dim> &p )
{
    std::cout << "Unsupported dim=" << dim << " in Spatial_mesh. Aborting.";
    exit( EXIT_FAILURE );
}

template<>
bool Domain<1>::out_of_bound( const Particle<1> &p )
{
    double x = p.position.x();
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 );
    return out;

}

template<>
bool Domain<2>::out_of_bound( const Particle<2> &p )
{
    double x = p.position.x();
    double y = p.position.y();
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= spat_mesh.y_volume_size ) || ( y <= 0 ) ;
    return out;

}

template<>
bool Domain<3>::out_of_bound( const Particle<3> &p )
{
    double x = p.position.x();
    double y = p.position.y();
    double z = p.position.z();
    bool out;
    
    out = 
	( x >= spat_mesh.x_volume_size ) || ( x <= 0 ) ||
	( y >= spat_mesh.y_volume_size ) || ( y <= 0 ) ||
	( z >= spat_mesh.z_volume_size ) || ( z <= 0 );
    return out;

}

template< int dim >
void Domain<dim>::generate_new_particles()
{
    particle_sources.generate_each_step();
    return;
}


//
// Update time grid
//

template< int dim >
void Domain<dim>::update_time_grid()
{
    time_grid.update_to_next_step();
    return;
}

//
// Write domain to file
//

template< int dim >
void Domain<dim>::write_step_to_save( Config<dim> &conf )
{
    int current_step = time_grid.current_node;
    int step_to_save = time_grid.node_to_save;
    if ( ( current_step % step_to_save ) == 0 ){	
	write( conf );
    }
    return;
}

template< int dim >
void Domain<dim>::write( Config<dim> &conf )
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
    spat_mesh.print( output_file );
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

template< int dim >
Domain<dim>::~Domain()
{
    std::cout << "TODO: free domain.\n";
    return;
}

//
// Various functions
//

template< int dim >
void Domain<dim>::print_particles() 
{
    particle_sources.print_particles();
    return;
}


#endif /* _DOMAIN_H_ */
