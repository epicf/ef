#ifndef _PARTICLE_SOURCE_H_
#define _PARTICLE_SOURCE_H_

#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Config.hpp"
#include "Particle.hpp"
#include "VecNd.hpp"



template< int dim > 
class Single_particle_source{
private:
    std::string name;
    //
    int initial_number_of_particles;
    int particles_to_generate_each_step;
    // Source position.     
    VecNd<dim> left_bottom_near;
    VecNd<dim> right_top_far;
    // Momentum
    VecNd<dim> mean_momentum;
    double temperature;
    // Particle characteristics
    double charge;
    double mass;
    // Random number generator
    std::default_random_engine rnd_gen;
public:
    std::vector< Particle<dim> > particles;
public:
    Single_particle_source( Config<dim> &conf, Source_config_part<dim> &src_conf  );
    void generate_each_step();
    void update_particles_position( double dt );	
    void print_particles();
    void write_to_file( std::ofstream &output_file );
    virtual ~Single_particle_source() {};
private:
    // Particle initialization
    void set_parameters_from_config( Source_config_part<dim> &src_conf );
    void generate_initial_particles();
    // Todo: replace 'std::default_random_engine' type with something more general.
    void generate_num_of_particles( int num_of_particles );
    int generate_particle_id( const int number );
    VecNd<dim> uniform_position_in_hyperrectangle( const VecNd<dim> left_bottom_near,
						   const VecNd<dim> right_top_far,
						   std::default_random_engine &rnd_gen );
    double random_in_range( const double low, const double up, std::default_random_engine &rnd_gen );
    VecNd<dim> maxwell_momentum_distr( const VecNd<dim> mean_momentum,
				       const double temperature, const double mass, 
				       std::default_random_engine &rnd_gen );
    // Check config
    void check_correctness_of_related_config_fields( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_initial_number_of_particles_ge_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_particles_to_generate_each_step_ge_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );

    void particle_source_x_left_ge_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_x_left_le_particle_source_x_right( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_x_right_le_grid_x_size( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_y_bottom_ge_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_y_bottom_le_particle_source_y_top( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_y_top_le_grid_y_size( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_z_near_ge_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_z_near_le_particle_source_z_far( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_z_far_le_grid_z_size( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );

    void particle_source_temperature_gt_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );
    void particle_source_mass_gt_zero( 
	Config<dim> &conf, Source_config_part<dim> &src_conf );

    void check_and_warn_if_not( const bool &should_be,
				const std::string &message );
};

template< int dim >
Single_particle_source<dim>::Single_particle_source( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_correctness_of_related_config_fields( conf, src_conf );
    set_parameters_from_config( src_conf );
    generate_initial_particles();
}

template< int dim >
void Single_particle_source<dim>::check_correctness_of_related_config_fields( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    particle_source_initial_number_of_particles_ge_zero( conf, src_conf );
    particle_source_particles_to_generate_each_step_ge_zero( conf, src_conf );
    if( dim < 1 || dim > 3 ){
	std::cout << "Unsupported dim=" << dim << " in Single_particle_source. Aborting.";
	exit( EXIT_FAILURE );
    }
    if( dim >=1 && dim <= 3 ){
	particle_source_x_left_ge_zero( conf, src_conf );
	particle_source_x_left_le_particle_source_x_right( conf, src_conf );
	particle_source_x_right_le_grid_x_size( conf, src_conf );	
    }
    if( dim >=2 && dim <= 3 ){
	particle_source_y_bottom_ge_zero( conf, src_conf );
	particle_source_y_bottom_le_particle_source_y_top( conf, src_conf );
	particle_source_y_top_le_grid_y_size( conf, src_conf );
    }
    if( dim == 3 ){
	particle_source_z_near_ge_zero( conf, src_conf );
	particle_source_z_near_le_particle_source_z_far( conf, src_conf );
	particle_source_z_far_le_grid_z_size( conf, src_conf );
    }
    particle_source_temperature_gt_zero( conf, src_conf );
    particle_source_mass_gt_zero( conf, src_conf );
}

template< int dim > 
void Single_particle_source<dim>::set_parameters_from_config( Source_config_part<dim> &src_conf )
{
    std::cout << "Unsupported dim=" << dim << " in Single_particle_source. Aborting.";
    exit( EXIT_FAILURE );
}

template<> 
void Single_particle_source<1>::set_parameters_from_config(
    Source_config_part<1> &src_conf )
{
    name = src_conf.particle_source_name;
    initial_number_of_particles = src_conf.particle_source_initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particle_source_particles_to_generate_each_step;

    left_bottom_near = VecNd<1>( src_conf.particle_source_x_left );
    right_top_far = VecNd<1>( src_conf.particle_source_x_right );
    mean_momentum = VecNd<1>( src_conf.particle_source_mean_momentum_x );

    temperature = src_conf.particle_source_temperature;
    charge = src_conf.particle_source_charge;
    mass = src_conf.particle_source_mass;    
    // Random number generator
    unsigned seed = 0;
    rnd_gen = std::default_random_engine( seed );
}

template<> 
void Single_particle_source<2>::set_parameters_from_config(
    Source_config_part<2> &src_conf )
{
    name = src_conf.particle_source_name;
    initial_number_of_particles = src_conf.particle_source_initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particle_source_particles_to_generate_each_step;

    left_bottom_near = VecNd<2>( src_conf.particle_source_x_left,
				 src_conf.particle_source_y_bottom );
    right_top_far = VecNd<2>( src_conf.particle_source_x_right,
			      src_conf.particle_source_y_top );
    mean_momentum = VecNd<2>( src_conf.particle_source_mean_momentum_x,
			      src_conf.particle_source_mean_momentum_y );

    temperature = src_conf.particle_source_temperature;
    charge = src_conf.particle_source_charge;
    mass = src_conf.particle_source_mass;    
    // Random number generator
    unsigned seed = 0;
    rnd_gen = std::default_random_engine( seed );
}

template<> 
void Single_particle_source<3>::set_parameters_from_config(
    Source_config_part<3> &src_conf )
{
    name = src_conf.particle_source_name;
    initial_number_of_particles = src_conf.particle_source_initial_number_of_particles;
    particles_to_generate_each_step = 
	src_conf.particle_source_particles_to_generate_each_step;

    left_bottom_near = VecNd<3>( src_conf.particle_source_x_left,
				 src_conf.particle_source_y_bottom,
				 src_conf.particle_source_z_near );
    right_top_far = VecNd<3>( src_conf.particle_source_x_right,
			      src_conf.particle_source_y_top,
			      src_conf.particle_source_z_far );
    mean_momentum = VecNd<3>( src_conf.particle_source_mean_momentum_x,
			      src_conf.particle_source_mean_momentum_y,
			      src_conf.particle_source_mean_momentum_z );

    temperature = src_conf.particle_source_temperature;
    charge = src_conf.particle_source_charge;
    mass = src_conf.particle_source_mass;    
    // Random number generator
    unsigned seed = 0;
    rnd_gen = std::default_random_engine( seed );
}

template< int dim >
void Single_particle_source<dim>::generate_initial_particles()
{
    //particles.reserve( initial_number_of_particles );
    generate_num_of_particles( initial_number_of_particles );
}

template< int dim >
void Single_particle_source<dim>::generate_each_step()
{
    //particles.reserve( particles.size() + particles_to_generate_each_step );
    generate_num_of_particles( particles_to_generate_each_step );
}

template< int dim >
void Single_particle_source<dim>::generate_num_of_particles( int num_of_particles )
{
    VecNd<dim> pos, mom;
    int id = 0;
                
    for ( int i = 0; i < num_of_particles; i++ ) {
	id = generate_particle_id( i );
	pos = uniform_position_in_hyperrectangle( left_bottom_near,
						  right_top_far,
						  rnd_gen );
	mom = maxwell_momentum_distr( mean_momentum, temperature, mass, rnd_gen );
	particles.emplace_back( id, charge, mass, pos, mom );
    }

}

template< int dim >
int Single_particle_source<dim>::generate_particle_id( const int number )
{    
    // Preserve max id between calls to generator.
    // Todo: create Max_id field in single_particle_source class
    static int last_id = 0;
    
    return last_id++;
}

template< int dim >
VecNd<dim> Single_particle_source<dim>::uniform_position_in_hyperrectangle( 
    const VecNd<dim> left_bottom_near,
    const VecNd<dim> right_top_far,
    std::default_random_engine &rnd_gen )
{
    std::cout << "Unsupported dim=" << dim << " in Single_particle_source. Aborting.";
    exit( EXIT_FAILURE );
  
}

template<>
VecNd<1> Single_particle_source<1>::uniform_position_in_hyperrectangle( 
    const VecNd<1> left_bottom_near,
    const VecNd<1> right_top_far,
    std::default_random_engine &rnd_gen )
{
    VecNd<1> pos;

    // todo: add construction from array to VecNd class and
    // rewrite this as loop over dim.
    pos = VecNd<1>( random_in_range( left_bottom_near.x(), right_top_far.x(), rnd_gen ) );
    
    return pos;
}

template<>
VecNd<2> Single_particle_source<2>::uniform_position_in_hyperrectangle( 
    const VecNd<2> left_bottom_near,
    const VecNd<2> right_top_far,
    std::default_random_engine &rnd_gen )
{
    VecNd<2> pos;

    // todo: add construction from array to VecNd class and
    // rewrite this as loop over dim.
    pos = VecNd<2>( random_in_range( left_bottom_near.x(), right_top_far.x(), rnd_gen ),
		    random_in_range( left_bottom_near.y(), right_top_far.y(), rnd_gen ) );
    
    return pos;
}

template<>
VecNd<3> Single_particle_source<3>::uniform_position_in_hyperrectangle( 
    const VecNd<3> left_bottom_near,
    const VecNd<3> right_top_far,
    std::default_random_engine &rnd_gen )
{
    VecNd<3> pos;

    // todo: add construction from array to VecNd class and
    // rewrite this as loop over dim.
    pos = VecNd<3>( random_in_range( left_bottom_near.x(), right_top_far.x(), rnd_gen ),
		    random_in_range( left_bottom_near.y(), right_top_far.y(), rnd_gen ),
		    random_in_range( left_bottom_near.z(), right_top_far.z(), rnd_gen ) );
  
    return pos;
}

template< int dim > 
double Single_particle_source<dim>::random_in_range( 
    const double low, const double up, 
    std::default_random_engine &rnd_gen )
{
    std::uniform_real_distribution<double> uniform_distr( low, up );
    return uniform_distr( rnd_gen );
}


template< int dim > 
VecNd<dim> Single_particle_source<dim>::maxwell_momentum_distr(
    const VecNd<dim> mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{
    std::cout << "Unsupported dim=" << dim << " in maxwell_momentum_distr. Aborting.";
    exit( EXIT_FAILURE );
}

template<> 
VecNd<1> Single_particle_source<1>::maxwell_momentum_distr(
    const VecNd<1> mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{
    VecNd<1> mom;
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    double maxwell_gauss_std_mean_x;
    // Resoursce acquisition is initialization.
    // Can't declare variables without calling constructor
    // std::normal_distribution<double> normal_distr_x, normal_distr_y, normal_distr_z;

    maxwell_gauss_std_mean_x = mean_momentum.x();
    std::normal_distribution<double>
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    mom = VecNd<1>( normal_distr_x( rnd_gen ) );
    
    mom = mom * 1.0; // recheck
    return mom;
}

template<> 
VecNd<2> Single_particle_source<2>::maxwell_momentum_distr(
    const VecNd<2> mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{
    VecNd<2> mom;
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    double maxwell_gauss_std_mean_x, maxwell_gauss_std_mean_y;
    // Resoursce acquisition is initialization.
    // Can't declare variables without calling constructor
    // std::normal_distribution<double> normal_distr_x, normal_distr_y, normal_distr_z;

    // todo: way too much repetition
    maxwell_gauss_std_mean_x = mean_momentum.x();
    maxwell_gauss_std_mean_y = mean_momentum.y();
    std::normal_distribution<double>
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    std::normal_distribution<double>
	normal_distr_y( maxwell_gauss_std_mean_y, maxwell_gauss_std_dev );
    mom = VecNd<2>( normal_distr_x( rnd_gen ),
		    normal_distr_y( rnd_gen ) );		     
    
    mom = mom * 1.0; // recheck
    return mom;
}

template<> 
VecNd<3> Single_particle_source<3>::maxwell_momentum_distr(
    const VecNd<3> mean_momentum, const double temperature, const double mass, 
    std::default_random_engine &rnd_gen )
{
    VecNd<3> mom;
    double maxwell_gauss_std_dev = sqrt( mass * temperature );
    double maxwell_gauss_std_mean_x, maxwell_gauss_std_mean_y, maxwell_gauss_std_mean_z;
    // Resoursce acquisition is initialization.
    // Can't declare variables without calling constructor
    // std::normal_distribution<double> normal_distr_x, normal_distr_y, normal_distr_z;

    maxwell_gauss_std_mean_x = mean_momentum.x();
    maxwell_gauss_std_mean_y = mean_momentum.y();
    maxwell_gauss_std_mean_z = mean_momentum.z();
    std::normal_distribution<double>
	normal_distr_x( maxwell_gauss_std_mean_x, maxwell_gauss_std_dev );
    std::normal_distribution<double>
	normal_distr_y( maxwell_gauss_std_mean_y, maxwell_gauss_std_dev );
    std::normal_distribution<double>
	normal_distr_z( maxwell_gauss_std_mean_z, maxwell_gauss_std_dev );

    mom = VecNd<3>( normal_distr_x( rnd_gen ),
		    normal_distr_y( rnd_gen ),
		    normal_distr_z( rnd_gen ) );    

    mom = mom * 1.0; // recheck
    return mom;
}


template< int dim >
void Single_particle_source<dim>::update_particles_position( double dt )
{
    for ( auto &p : particles )
	p.update_position( dt );
}

template< int dim >
void Single_particle_source<dim>::print_particles()
{
    std::cout << "Source name: " << name << std::endl;
    for ( auto& p : particles  ) {	
	p.print_short();
    }
    return;
}

template< int dim >
void Single_particle_source<dim>::write_to_file( std::ofstream &output_file )
{
    std::cout << "Writing Source name = " << name << ", "
	      << "number of particles = " << particles.size() 
	      << std::endl;
    output_file << "Source name = " << name << std::endl;
    output_file << "Total number of particles = " << particles.size() << std::endl;
    output_file << "id, charge, mass, position, momentum" << std::endl;
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 3 );    
    output_file.setf( std::ios::right );
    for ( auto &p : particles ) {	
	output_file << std::setw(10) << std::left << p.id
		    << std::setw(15) << p.charge
		    << std::setw(15) << p.mass
		    << std::setw(25) << p.position
		    << std::setw(25) << p.momentum
		    << std::endl;
    }
    return;
}

template< int dim >
void Single_particle_source<dim>::particle_source_initial_number_of_particles_ge_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_initial_number_of_particles >= 0,
	"particle_source_initial_number_of_particles <= 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_particles_to_generate_each_step_ge_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_particles_to_generate_each_step >= 0,
	"particle_source_particles_to_generate_each_step < 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_x_left_ge_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left >= 0,
	"particle_source_x_left < 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_x_left_le_particle_source_x_right( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_left <= src_conf.particle_source_x_right,
	"particle_source_x_left > particle_source_x_right" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_x_right_le_grid_x_size( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_x_right <= conf.mesh_config_part.grid_x_size,
	"particle_source_x_right > grid_x_size" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_y_bottom_ge_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom >= 0,
	"particle_source_y_bottom < 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_y_bottom_le_particle_source_y_top( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_bottom <= src_conf.particle_source_y_top,
	"particle_source_y_bottom > particle_source_y_top" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_y_top_le_grid_y_size( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_y_top <= conf.mesh_config_part.grid_y_size,
	"particle_source_y_top > grid_y_size" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_z_near_ge_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near >= 0,
	"particle_source_z_near < 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_z_near_le_particle_source_z_far( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_near <= src_conf.particle_source_z_far,
	"particle_source_z_near > particle_source_z_far" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_z_far_le_grid_z_size( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_z_far <= conf.mesh_config_part.grid_z_size,
	"particle_source_z_far > grid_z_size" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_temperature_gt_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_temperature >= 0,
	"particle_source_temperature < 0" );
}

template< int dim >
void Single_particle_source<dim>::particle_source_mass_gt_zero( 
    Config<dim> &conf, 
    Source_config_part<dim> &src_conf )
{
    check_and_warn_if_not( 
	src_conf.particle_source_mass >= 0,
	"particle_source_mass < 0" );
}


template< int dim >
void Single_particle_source<dim>::check_and_warn_if_not( const bool &should_be,
							 const std::string &message )
{
    if( !should_be ){
	std::cout << "Warning: " + message << std::endl;
    }
    return;
}


template< int dim >
class Particle_sources{
public:
    std::vector< Single_particle_source<dim> > sources;
public:
    Particle_sources( Config<dim> &conf );
    virtual ~Particle_sources() {};
    void write_to_file( std::ofstream &output_file ) 
    {
	output_file << "### Particles" << std::endl;
	for( auto &src : sources )
	    src.write_to_file( output_file );
    }
    void generate_each_step()
    {
	for( auto &src : sources )
	    src.generate_each_step();
    }
    void print_particles()
    {
	for( auto &src : sources )
	    src.print_particles();
    }

    void update_particles_position( double dt )
    {
	for( auto &src : sources )
	    src.update_particles_position( dt );
    }
};

template< int dim >
Particle_sources<dim>::Particle_sources( Config<dim> &conf )
{
    for( auto &src_conf : conf.sources_config_part ) {
	sources.emplace_back( conf, src_conf );
    }
}


#endif /* _PARTICLE_SOURCE_H_ */
