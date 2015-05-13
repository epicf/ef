#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <iostream>
#include <string>
#include <limits>

// Todo:: rewrite with inheritance.
// Add some macroprogramming or templates. Too much similar code.

class Time_config_part{
public:
    double total_time;
    double time_step_size;
    double time_save_step;
public:
    Time_config_part(){};
    Time_config_part( boost::property_tree::ptree &ptree ) :
	total_time( ptree.get<double>("total_time") ),
	time_step_size( ptree.get<double>("time_step_size") ),
	time_save_step( ptree.get<double>("time_save_step") )
	{} ;
    virtual ~Time_config_part() {};
    void print() {
	std::cout << "Total_time = " << total_time << std::endl;
	std::cout << "Time_step_size = " << time_step_size << std::endl;
	std::cout << "Time_save_step = " << time_save_step << std::endl;
    }
};

template< int dim >
class Mesh_config_part{
public:
public:
    Mesh_config_part(){};
    Mesh_config_part( boost::property_tree::ptree &ptree ){
	std::cout << "Unsupported dimensionality. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    };
    virtual ~Mesh_config_part() {};
    void print() {};
};

template<>
class Mesh_config_part<1>{
public:
    double grid_x_size;
    double grid_x_step;
public:
    Mesh_config_part(){};
    Mesh_config_part( boost::property_tree::ptree &ptree ) :
	grid_x_size( ptree.get<double>("grid_x_size") ),
	grid_x_step( ptree.get<double>("grid_x_step") )
	{};
    virtual ~Mesh_config_part() {};
    void print() {
	std::cout << "grid_x_size = " << grid_x_size << std::endl;
	std::cout << "grid_x_step = " << grid_x_step << std::endl;
    }
};


template<>
class Mesh_config_part<2>{
public:
    double grid_x_size;
    double grid_x_step;
    double grid_y_size;
    double grid_y_step;
public:
    Mesh_config_part(){};
    Mesh_config_part( boost::property_tree::ptree &ptree ) :
	grid_x_size( ptree.get<double>("grid_x_size") ),
	grid_x_step( ptree.get<double>("grid_x_step") ),
        grid_y_size( ptree.get<double>("grid_y_size") ),
	grid_y_step( ptree.get<double>("grid_y_step") )
	{};
    virtual ~Mesh_config_part() {};
    void print() {
	std::cout << "grid_x_size = " << grid_x_size << std::endl;
	std::cout << "grid_x_step = " << grid_x_step << std::endl;
	std::cout << "grid_y_size = " << grid_y_size << std::endl;
	std::cout << "grid_y_step = " << grid_y_step << std::endl;
    }
};

template<>
class Mesh_config_part<3>{
public:
    double grid_x_size;
    double grid_x_step;
    double grid_y_size;
    double grid_y_step;
    double grid_z_size;
    double grid_z_step;
public:
    Mesh_config_part(){};
    Mesh_config_part( boost::property_tree::ptree &ptree ) :
	grid_x_size( ptree.get<double>("grid_x_size") ),
	grid_x_step( ptree.get<double>("grid_x_step") ),
        grid_y_size( ptree.get<double>("grid_y_size") ),
	grid_y_step( ptree.get<double>("grid_y_step") ),
	grid_z_size( ptree.get<double>("grid_z_size") ),
	grid_z_step( ptree.get<double>("grid_z_step") )
	{};
    virtual ~Mesh_config_part() {};
    void print() {
	std::cout << "grid_x_size = " << grid_x_size << std::endl;
	std::cout << "grid_x_step = " << grid_x_step << std::endl;
	std::cout << "grid_y_size = " << grid_y_size << std::endl;
	std::cout << "grid_y_step = " << grid_y_step << std::endl;
	std::cout << "grid_z_size = " << grid_z_size << std::endl;
	std::cout << "grid_z_step = " << grid_z_step << std::endl;
    }
};


template< int dim >
class Source_config_part{
public:
public:
    Source_config_part(){};
    Source_config_part( std::string name, boost::property_tree::ptree &ptree ){
	std::cout << "Unsupported dimensionality. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    };
    virtual ~Source_config_part() {};
    void print() {};
};

template<>
class Source_config_part<1>{
public:
    std::string particle_source_name;
    int particle_source_initial_number_of_particles;
    int particle_source_particles_to_generate_each_step;
    double particle_source_x_left;
    double particle_source_x_right;
    double particle_source_mean_momentum_x;
    double particle_source_temperature;
    double particle_source_charge;
    double particle_source_mass;
public:
    Source_config_part(){};
    Source_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	particle_source_name( name ),
	particle_source_initial_number_of_particles(
	    ptree.get<int>("particle_source_initial_number_of_particles") ),
	particle_source_particles_to_generate_each_step(
	    ptree.get<int>("particle_source_particles_to_generate_each_step") ),
	particle_source_x_left( ptree.get<double>("particle_source_x_left") ),
	particle_source_x_right( ptree.get<double>("particle_source_x_right") ),
	particle_source_mean_momentum_x( ptree.get<double>("particle_source_mean_momentum_x") ),
	particle_source_temperature( ptree.get<double>("particle_source_temperature") ),
        particle_source_charge( ptree.get<double>("particle_source_charge") ),
	particle_source_mass( ptree.get<double>("particle_source_mass") )
	{};
    virtual ~Source_config_part() {};
    void print() { 
	std::cout << "Source: name = " << particle_source_name << std::endl;
	std::cout << "particle_source_initial_number_of_particles = " << 
	    particle_source_initial_number_of_particles << std::endl; 
	std::cout << "particles_to_generate_each_step = " << 
	    particle_source_particles_to_generate_each_step << std::endl; 
	std::cout << "particle_source_x_left = " << particle_source_x_left << std::endl;
	std::cout << "particle_source_x_right = " << particle_source_x_right << std::endl;
	std::cout << "particle_source_mean_momentum_x = "
		  << particle_source_mean_momentum_x << std::endl;
	std::cout << "particle_source_temperature = " << particle_source_temperature << std::endl;
	std::cout << "particle_source_charge = " << particle_source_charge << std::endl;
	std::cout << "particle_source_mass = " << particle_source_mass << std::endl;
    }
};

template<>
class Source_config_part<2>{
public:
    std::string particle_source_name;
    int particle_source_initial_number_of_particles;
    int particle_source_particles_to_generate_each_step;
    double particle_source_x_left;
    double particle_source_x_right;
    double particle_source_y_bottom;
    double particle_source_y_top;
    double particle_source_mean_momentum_x;
    double particle_source_mean_momentum_y;
    double particle_source_temperature;
    double particle_source_charge;
    double particle_source_mass;
public:
    Source_config_part(){};
    Source_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	particle_source_name( name ),
	particle_source_initial_number_of_particles(
	    ptree.get<int>("particle_source_initial_number_of_particles") ),
	particle_source_particles_to_generate_each_step(
	    ptree.get<int>("particle_source_particles_to_generate_each_step") ),
	particle_source_x_left( ptree.get<double>("particle_source_x_left") ),
	particle_source_x_right( ptree.get<double>("particle_source_x_right") ),
        particle_source_y_bottom( ptree.get<double>("particle_source_y_bottom") ),
	particle_source_y_top( ptree.get<double>("particle_source_y_top") ),
	particle_source_mean_momentum_x( ptree.get<double>("particle_source_mean_momentum_x") ),
	particle_source_mean_momentum_y( ptree.get<double>("particle_source_mean_momentum_y") ),
	particle_source_temperature( ptree.get<double>("particle_source_temperature") ),
        particle_source_charge( ptree.get<double>("particle_source_charge") ),
	particle_source_mass( ptree.get<double>("particle_source_mass") )
	{};
    virtual ~Source_config_part() {};
    void print() { 
	std::cout << "Source: name = " << particle_source_name << std::endl;
	std::cout << "particle_source_initial_number_of_particles = " << 
	    particle_source_initial_number_of_particles << std::endl; 
	std::cout << "particles_to_generate_each_step = " << 
	    particle_source_particles_to_generate_each_step << std::endl; 
	std::cout << "particle_source_x_left = " << particle_source_x_left << std::endl;
	std::cout << "particle_source_x_right = " << particle_source_x_right << std::endl;
	std::cout << "particle_source_y_bottom = " << particle_source_y_bottom << std::endl;
	std::cout << "particle_source_y_top = " << particle_source_y_top << std::endl;
	std::cout << "particle_source_mean_momentum_x = "
		  << particle_source_mean_momentum_x << std::endl;
	std::cout << "particle_source_mean_momentum_y = "
		  << particle_source_mean_momentum_y << std::endl;
	std::cout << "particle_source_temperature = " << particle_source_temperature << std::endl;
	std::cout << "particle_source_charge = " << particle_source_charge << std::endl;
	std::cout << "particle_source_mass = " << particle_source_mass << std::endl;
    }
};

template<>
class Source_config_part<3>{
public:
    std::string particle_source_name;
    int particle_source_initial_number_of_particles;
    int particle_source_particles_to_generate_each_step;
    double particle_source_x_left;
    double particle_source_x_right;
    double particle_source_y_bottom;
    double particle_source_y_top;
    double particle_source_z_near;
    double particle_source_z_far;
    double particle_source_mean_momentum_x;
    double particle_source_mean_momentum_y;
    double particle_source_mean_momentum_z;
    double particle_source_temperature;
    double particle_source_charge;
    double particle_source_mass;
public:
    Source_config_part(){};
    Source_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	particle_source_name( name ),
	particle_source_initial_number_of_particles(
	    ptree.get<int>("particle_source_initial_number_of_particles") ),
	particle_source_particles_to_generate_each_step(
	    ptree.get<int>("particle_source_particles_to_generate_each_step") ),
	particle_source_x_left( ptree.get<double>("particle_source_x_left") ),
	particle_source_x_right( ptree.get<double>("particle_source_x_right") ),
        particle_source_y_bottom( ptree.get<double>("particle_source_y_bottom") ),
	particle_source_y_top( ptree.get<double>("particle_source_y_top") ),
	particle_source_z_near( ptree.get<double>("particle_source_z_near") ),
	particle_source_z_far( ptree.get<double>("particle_source_z_far") ),
	particle_source_mean_momentum_x( ptree.get<double>("particle_source_mean_momentum_x") ),
	particle_source_mean_momentum_y( ptree.get<double>("particle_source_mean_momentum_y") ),
	particle_source_mean_momentum_z( ptree.get<double>("particle_source_mean_momentum_z") ),
	particle_source_temperature( ptree.get<double>("particle_source_temperature") ),
        particle_source_charge( ptree.get<double>("particle_source_charge") ),
	particle_source_mass( ptree.get<double>("particle_source_mass") )
	{};
    virtual ~Source_config_part() {};
    void print() { 
	std::cout << "Source: name = " << particle_source_name << std::endl;
	std::cout << "particle_source_initial_number_of_particles = " << 
	    particle_source_initial_number_of_particles << std::endl; 
	std::cout << "particles_to_generate_each_step = " << 
	    particle_source_particles_to_generate_each_step << std::endl; 
	std::cout << "particle_source_x_left = " << particle_source_x_left << std::endl;
	std::cout << "particle_source_x_right = " << particle_source_x_right << std::endl;
	std::cout << "particle_source_y_bottom = " << particle_source_y_bottom << std::endl;
	std::cout << "particle_source_y_top = " << particle_source_y_top << std::endl;
	std::cout << "particle_source_z_near = " << particle_source_z_near << std::endl;
	std::cout << "particle_source_z_far = " << particle_source_z_far << std::endl;
	std::cout << "particle_source_mean_momentum_x = "
		  << particle_source_mean_momentum_x << std::endl;
	std::cout << "particle_source_mean_momentum_y = "
		  << particle_source_mean_momentum_y << std::endl;
	std::cout << "particle_source_mean_momentum_z = "
		  << particle_source_mean_momentum_z << std::endl;
	std::cout << "particle_source_temperature = " << particle_source_temperature << std::endl;
	std::cout << "particle_source_charge = " << particle_source_charge << std::endl;
	std::cout << "particle_source_mass = " << particle_source_mass << std::endl;
    }
};


template< int dim >
class Inner_region_config_part{
public:
    Inner_region_config_part(){};
    Inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ){
	std::cout << "Unsupported dimensionality. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    };
    virtual ~Inner_region_config_part() {};
    void print() {};
};

template<>
class Inner_region_config_part<1>{
public:
    std::string inner_region_name;
    double inner_region_x_left;
    double inner_region_x_right;
    double inner_region_boundary_potential;
public:
    Inner_region_config_part(){};
    Inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_name( name ),
	inner_region_x_left( ptree.get<double>("inner_region_x_left") ),
	inner_region_x_right( ptree.get<double>("inner_region_x_right") ),
	inner_region_boundary_potential( ptree.get<double>("inner_region_boundary_potential") )
	{};
    virtual ~Inner_region_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << inner_region_name << std::endl;
	std::cout << "inner_region_x_left = " << inner_region_x_left << std::endl;
	std::cout << "inner_region_x_right = " << inner_region_x_right << std::endl;
	std::cout << "inner_region_boundary_potential = "
		  << inner_region_boundary_potential << std::endl;
    }
};


template<>
class Inner_region_config_part<2>{
public:
    std::string inner_region_name;
    double inner_region_x_left;
    double inner_region_x_right;
    double inner_region_y_bottom;
    double inner_region_y_top;
    double inner_region_boundary_potential;
public:
    Inner_region_config_part(){};
    Inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_name( name ),
	inner_region_x_left( ptree.get<double>("inner_region_x_left") ),
	inner_region_x_right( ptree.get<double>("inner_region_x_right") ),
        inner_region_y_bottom( ptree.get<double>("inner_region_y_bottom") ),
	inner_region_y_top( ptree.get<double>("inner_region_y_top") ),
	inner_region_boundary_potential( ptree.get<double>("inner_region_boundary_potential") )
	{};
    virtual ~Inner_region_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << inner_region_name << std::endl;
	std::cout << "inner_region_x_left = " << inner_region_x_left << std::endl;
	std::cout << "inner_region_x_right = " << inner_region_x_right << std::endl;
	std::cout << "inner_region_y_bottom = " << inner_region_y_bottom << std::endl;
	std::cout << "inner_region_y_top = " << inner_region_y_top << std::endl;
	std::cout << "inner_region_boundary_potential = "
		  << inner_region_boundary_potential << std::endl;
    }
};

template<>
class Inner_region_config_part<3>{
public:
    std::string inner_region_name;
    double inner_region_x_left;
    double inner_region_x_right;
    double inner_region_y_bottom;
    double inner_region_y_top;
    double inner_region_z_near;
    double inner_region_z_far;
    double inner_region_boundary_potential;
public:
    Inner_region_config_part(){};
    Inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_name( name ),
	inner_region_x_left( ptree.get<double>("inner_region_x_left") ),
	inner_region_x_right( ptree.get<double>("inner_region_x_right") ),
        inner_region_y_bottom( ptree.get<double>("inner_region_y_bottom") ),
	inner_region_y_top( ptree.get<double>("inner_region_y_top") ),
	inner_region_z_near( ptree.get<double>("inner_region_z_near") ),
	inner_region_z_far( ptree.get<double>("inner_region_z_far") ),
	inner_region_boundary_potential( ptree.get<double>("inner_region_boundary_potential") )
	{};
    virtual ~Inner_region_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << inner_region_name << std::endl;
	std::cout << "inner_region_x_left = " << inner_region_x_left << std::endl;
	std::cout << "inner_region_x_right = " << inner_region_x_right << std::endl;
	std::cout << "inner_region_y_bottom = " << inner_region_y_bottom << std::endl;
	std::cout << "inner_region_y_top = " << inner_region_y_top << std::endl;
	std::cout << "inner_region_z_near = " << inner_region_z_near << std::endl;
	std::cout << "inner_region_z_far = " << inner_region_z_far << std::endl;
	std::cout << "inner_region_boundary_potential = "
		  << inner_region_boundary_potential << std::endl;
    }
};


template< int dim >
class Boundary_config_part {
public:
    Boundary_config_part(){};
    Boundary_config_part( boost::property_tree::ptree &ptree ){
	std::cout << "Unsupported dimensionality. Aborting" << std::endl;
	exit( EXIT_FAILURE );
    };
    virtual ~Boundary_config_part() {};
    void print() {};
};

template<>
class Boundary_config_part<1> {
public:
    double boundary_phi_left;
    double boundary_phi_right;
public:
    Boundary_config_part(){};
    Boundary_config_part( boost::property_tree::ptree &ptree ) :
	boundary_phi_left( ptree.get<double>("boundary_phi_left") ),
	boundary_phi_right( ptree.get<double>("boundary_phi_right") )
	{} ;
    virtual ~Boundary_config_part() {};
    void print() {
	std::cout << "boundary_phi_left = " << boundary_phi_left << std::endl;
	std::cout << "boundary_phi_right = " << boundary_phi_right << std::endl;
    }
};

template<>
class Boundary_config_part<2> {
public:
    double boundary_phi_left;
    double boundary_phi_right;
    double boundary_phi_bottom;
    double boundary_phi_top;
public:
    Boundary_config_part(){};
    Boundary_config_part( boost::property_tree::ptree &ptree ) :
	boundary_phi_left( ptree.get<double>("boundary_phi_left") ),
	boundary_phi_right( ptree.get<double>("boundary_phi_right") ),
        boundary_phi_bottom( ptree.get<double>("boundary_phi_bottom") ),
	boundary_phi_top( ptree.get<double>("boundary_phi_top") )
	{} ;
    virtual ~Boundary_config_part() {};
    void print() {
	std::cout << "boundary_phi_left = " << boundary_phi_left << std::endl;
	std::cout << "boundary_phi_right = " << boundary_phi_right << std::endl;
	std::cout << "boundary_phi_bottom = " << boundary_phi_bottom << std::endl;
	std::cout << "boundary_phi_top = " << boundary_phi_top << std::endl;
    }
};

template<>
class Boundary_config_part<3> {
public:
    double boundary_phi_left;
    double boundary_phi_right;
    double boundary_phi_bottom;
    double boundary_phi_top;
    double boundary_phi_near;
    double boundary_phi_far;
public:
    Boundary_config_part(){};
    Boundary_config_part( boost::property_tree::ptree &ptree ) :
	boundary_phi_left( ptree.get<double>("boundary_phi_left") ),
	boundary_phi_right( ptree.get<double>("boundary_phi_right") ),
        boundary_phi_bottom( ptree.get<double>("boundary_phi_bottom") ),
	boundary_phi_top( ptree.get<double>("boundary_phi_top") ),
	boundary_phi_near( ptree.get<double>("boundary_phi_near") ),
	boundary_phi_far( ptree.get<double>("boundary_phi_far") )
	{} ;
    virtual ~Boundary_config_part() {};
    void print() {
	std::cout << "boundary_phi_left = " << boundary_phi_left << std::endl;
	std::cout << "boundary_phi_right = " << boundary_phi_right << std::endl;
	std::cout << "boundary_phi_bottom = " << boundary_phi_bottom << std::endl;
	std::cout << "boundary_phi_top = " << boundary_phi_top << std::endl;
	std::cout << "boundary_phi_near = " << boundary_phi_near << std::endl;
	std::cout << "boundary_phi_far = " << boundary_phi_far << std::endl;
    }
};

class Output_filename_config_part {
public:
    std::string output_filename_prefix;
    std::string output_filename_suffix;
public:
    Output_filename_config_part(){};
    Output_filename_config_part( boost::property_tree::ptree &ptree ) :
	output_filename_prefix( ptree.get<std::string>("output_filename_prefix") ),
	output_filename_suffix( ptree.get<std::string>("output_filename_suffix") )
	{} ;
    virtual ~Output_filename_config_part() {};
    void print() {
	std::cout << "Output_filename_prefix = " << output_filename_prefix << std::endl;
	std::cout << "Output_filename_suffix = " << output_filename_suffix << std::endl;
    }
};

template< int dim >
class Config {
public:
    Time_config_part time_config_part;
    Mesh_config_part<dim> mesh_config_part;
    std::vector< Source_config_part<dim> > sources_config_part;
    std::vector< Inner_region_config_part<dim> > inner_regions_config_part;
    Boundary_config_part<dim> boundary_config_part;
    Output_filename_config_part output_filename_config_part;
public:
    Config( const std::string &filename );
    virtual ~Config() {};
    void print() { 
	std::cout << "=== Config file echo ===" << std::endl;
	time_config_part.print();
	mesh_config_part.print();
	for ( auto &s : sources_config_part ) {
	    s.print();
	}
	for ( auto &ir : inner_regions_config_part ) {
	    ir.print();
	}
	boundary_config_part.print();
	output_filename_config_part.print();
	std::cout << "======" << std::endl;
    }
};

template< int dim >
Config<dim>::Config( const std::string &filename )
{    
    try {
	using boost::property_tree::ptree;
	ptree pt;

	read_ini(filename, pt);
	
	for( auto &sections : pt ) {
	    std::string section_name = sections.first.data();
	    if ( section_name.find( "Time grid" ) != std::string::npos ) {
		time_config_part = Time_config_part( sections.second );
	    } else if ( section_name.find( "Spatial mesh" ) != std::string::npos ) {		
		mesh_config_part = Mesh_config_part<dim>( sections.second );
	    } else if ( section_name.find( "Source." ) != std::string::npos ) {
		std::string source_name = section_name.substr( section_name.find(".") + 1 );
		sources_config_part.emplace_back( source_name, sections.second );
	    } else if ( section_name.find( "Inner_region." ) != std::string::npos ) {
		std::string inner_region_name = section_name.substr( section_name.find(".") + 1 );
		inner_regions_config_part.emplace_back( inner_region_name, sections.second );
	    } else if ( section_name.find( "Boundary conditions" ) != std::string::npos ) {
		boundary_config_part = Boundary_config_part<dim>( sections.second );
	    } else if ( section_name.find( "Output filename" ) != std::string::npos ) {
		output_filename_config_part = Output_filename_config_part( sections.second );				
	    } else {
		std::cout << "Ignoring unknown section: " << section_name << std::endl;
	    }
	}
	return;
    }
    catch( std::exception& e ) {
        std::cerr << "error: " << e.what() << "\n";
        exit( EXIT_FAILURE);
    }
    catch( ... ) {
        std::cerr << "Exception of unknown type!\n";
    }
}


#endif /* _CONFIG_H_ */
