#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <iostream>
#include <string>
#include <limits>

// Todo:: rewrite with inheritance.
// Add some macroprogramming or templates.
// Too much similar code.

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

class Mesh_config_part{
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

class Source_config_part{
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
	particle_source_initial_number_of_particles( ptree.get<int>("particle_source_initial_number_of_particles") ),
	particle_source_particles_to_generate_each_step( ptree.get<int>("particle_source_particles_to_generate_each_step") ),
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
	std::cout << "particle_source_mean_momentum_x = " << particle_source_mean_momentum_x << std::endl;
	std::cout << "particle_source_mean_momentum_y = " << particle_source_mean_momentum_y << std::endl;
	std::cout << "particle_source_mean_momentum_z = " << particle_source_mean_momentum_z << std::endl;
	std::cout << "particle_source_temperature = " << particle_source_temperature << std::endl;
	std::cout << "particle_source_charge = " << particle_source_charge << std::endl;
	std::cout << "particle_source_mass = " << particle_source_mass << std::endl;
    }
};


class Inner_region_config_part {
public:
    std::string inner_region_name;
    double inner_region_potential;
public:
    Inner_region_config_part(){};
    Inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_name( name ),
	inner_region_potential( ptree.get<double>("inner_region_potential") )
	{};	
    virtual ~Inner_region_config_part() {};
    virtual void print() {
	std::cout << "Inner_region_name = " << inner_region_name << std::endl;
	std::cout << "Inner_region_potential = " << inner_region_potential << std::endl;
    }
};

class Inner_region_box_config_part : public Inner_region_config_part{
public:
    double inner_region_box_x_left;
    double inner_region_box_x_right;
    double inner_region_box_y_bottom;
    double inner_region_box_y_top;
    double inner_region_box_z_near;
    double inner_region_box_z_far;
public:
    Inner_region_box_config_part(){};
    Inner_region_box_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_box_x_left( ptree.get<double>("inner_region_box_x_left") ),
	inner_region_box_x_right( ptree.get<double>("inner_region_box_x_right") ),
        inner_region_box_y_bottom( ptree.get<double>("inner_region_box_y_bottom") ),
	inner_region_box_y_top( ptree.get<double>("inner_region_box_y_top") ),
	inner_region_box_z_near( ptree.get<double>("inner_region_box_z_near") ),
	inner_region_box_z_far( ptree.get<double>("inner_region_box_z_far") ) {
	    inner_region_name = name;
	    inner_region_potential = ptree.get<double>("inner_region_box_potential");
	};
    virtual ~Inner_region_box_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << inner_region_name << std::endl;
	std::cout << "inner_region_potential = " << inner_region_potential << std::endl;
	std::cout << "inner_region_box_x_left = " << inner_region_box_x_left << std::endl;
	std::cout << "inner_region_box_x_right = " << inner_region_box_x_right << std::endl;
	std::cout << "inner_region_box_y_bottom = " << inner_region_box_y_bottom << std::endl;
	std::cout << "inner_region_box_y_top = " << inner_region_box_y_top << std::endl;
	std::cout << "inner_region_box_z_near = " << inner_region_box_z_near << std::endl;
	std::cout << "inner_region_box_z_far = " << inner_region_box_z_far << std::endl;
    }
};

class Inner_region_sphere_config_part : public Inner_region_config_part{
public:
    double inner_region_sphere_origin_x;
    double inner_region_sphere_origin_y;
    double inner_region_sphere_origin_z;
    double inner_region_sphere_radius;
public:
    Inner_region_sphere_config_part(){};
    Inner_region_sphere_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	inner_region_sphere_origin_x( ptree.get<double>("inner_region_sphere_origin_x") ),
	inner_region_sphere_origin_y( ptree.get<double>("inner_region_sphere_origin_y") ),
	inner_region_sphere_origin_z( ptree.get<double>("inner_region_sphere_origin_z") ),
	inner_region_sphere_radius( ptree.get<double>("inner_region_sphere_radius") ){
	    inner_region_name = name;
	    inner_region_potential = ptree.get<double>("inner_region_sphere_potential");
	};
    virtual ~Inner_region_sphere_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << inner_region_name << std::endl;
	std::cout << "inner_region_potential = " << inner_region_potential << std::endl;
	std::cout << "inner_region_sphere_origin_x = " << inner_region_sphere_origin_x << std::endl;
	std::cout << "inner_region_sphere_origin_y = " << inner_region_sphere_origin_y << std::endl;
	std::cout << "inner_region_sphere_origin_z = " << inner_region_sphere_origin_z << std::endl;
	std::cout << "inner_region_sphere_radius = " << inner_region_sphere_radius << std::endl;
    }
};

class Inner_region_STEP_config_part : public Inner_region_config_part {
  public:
    std::string inner_region_STEP_file;
  public:
    Inner_region_STEP_config_part(){};
  Inner_region_STEP_config_part( std::string name, boost::property_tree::ptree &ptree ) :
    inner_region_STEP_file( ptree.get<std::string>("inner_region_STEP_file") ) {
	inner_region_name = name;
	inner_region_potential = ptree.get<double>("inner_region_STEP_potential");
    };
    virtual ~Inner_region_STEP_config_part() {};
    void print() {
	std::cout << "Inner_region_STEP_name = " << inner_region_name << std::endl;
	std::cout << "Inner_region_STEP_potential = " << inner_region_potential << std::endl;
	std::cout << "Inner_region_STEP_file = " << inner_region_STEP_file << std::endl;
    }
};


class Charged_inner_region_config_part {
public:
    std::string charged_inner_region_name;
    double charged_inner_region_charge_density;
public:
    Charged_inner_region_config_part(){};
    Charged_inner_region_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	charged_inner_region_name( name ),
	charged_inner_region_charge_density( ptree.get<double>("charged_inner_region_charge_density") )
	{};	
    virtual ~Charged_inner_region_config_part() {};
    virtual void print() {
	std::cout << "Charged_inner_region_name = "
		  << charged_inner_region_name << std::endl;
	std::cout << "Charged_inner_region_charge_density = "
		  << charged_inner_region_charge_density << std::endl;
    }
};

class Charged_inner_region_box_config_part : public Charged_inner_region_config_part{
public:
    double charged_inner_region_box_x_left;
    double charged_inner_region_box_x_right;
    double charged_inner_region_box_y_bottom;
    double charged_inner_region_box_y_top;
    double charged_inner_region_box_z_near;
    double charged_inner_region_box_z_far;
public:
    Charged_inner_region_box_config_part(){};
    Charged_inner_region_box_config_part( std::string name,
					  boost::property_tree::ptree &ptree ) :
	charged_inner_region_box_x_left(
	    ptree.get<double>("charged_inner_region_box_x_left") ),
	charged_inner_region_box_x_right(
	    ptree.get<double>("charged_inner_region_box_x_right") ),
        charged_inner_region_box_y_bottom(
	    ptree.get<double>("charged_inner_region_box_y_bottom") ),
	charged_inner_region_box_y_top(
	    ptree.get<double>("charged_inner_region_box_y_top") ),
	charged_inner_region_box_z_near(
	    ptree.get<double>("charged_inner_region_box_z_near") ),
	charged_inner_region_box_z_far(
	    ptree.get<double>("charged_inner_region_box_z_far") ) {
	    charged_inner_region_name = name;
	    charged_inner_region_charge_density =
		ptree.get<double>("charged_inner_region_box_charge_density");
	};
    virtual ~Charged_inner_region_box_config_part() {};
    void print() { 
	std::cout << "Charged inner region: name = "
		  << charged_inner_region_name << std::endl;
	std::cout << "charged_inner_region_charge_density = "
		  << charged_inner_region_charge_density << std::endl;
	std::cout << "charged_inner_region_box_x_left = "
		  << charged_inner_region_box_x_left << std::endl;
	std::cout << "charged_inner_region_box_x_right = "
		  << charged_inner_region_box_x_right << std::endl;
	std::cout << "charged_inner_region_box_y_bottom = "
		  << charged_inner_region_box_y_bottom << std::endl;
	std::cout << "charged_inner_region_box_y_top = "
		  << charged_inner_region_box_y_top << std::endl;
	std::cout << "charged_inner_region_box_z_near = "
		  << charged_inner_region_box_z_near << std::endl;
	std::cout << "charged_inner_region_box_z_far = "
		  << charged_inner_region_box_z_far << std::endl;
    }
};

class Boundary_config_part {
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

class External_magnetic_field_config_part {
public:
    double magnetic_field_x;
    double magnetic_field_y;
    double magnetic_field_z;
    double speed_of_light;
public:
    External_magnetic_field_config_part(){};
    External_magnetic_field_config_part( boost::property_tree::ptree &ptree ) :
	magnetic_field_x( ptree.get<double>("magnetic_field_x") ),
	magnetic_field_y( ptree.get<double>("magnetic_field_y") ),
	magnetic_field_z( ptree.get<double>("magnetic_field_z") ),
	speed_of_light( ptree.get<double>("speed_of_light") )
	{} ;
    virtual ~External_magnetic_field_config_part() {};
    void print() {
	std::cout << "magnetic_field_x = " << magnetic_field_x << std::endl;
	std::cout << "magnetic_field_y = " << magnetic_field_y << std::endl;
	std::cout << "magnetic_field_z = " << magnetic_field_z << std::endl;
	std::cout << "speed_of_light = " << speed_of_light << std::endl;
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

class Config {
public:
    Time_config_part time_config_part;
    Mesh_config_part mesh_config_part;
    std::vector<Source_config_part> sources_config_part;
    boost::ptr_vector<Inner_region_config_part> inner_regions_config_part;
    boost::ptr_vector<Charged_inner_region_config_part> charged_inner_regions_config_part;
    Boundary_config_part boundary_config_part;
    External_magnetic_field_config_part external_magnetic_field_config_part;
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
	for ( auto &cir : charged_inner_regions_config_part ) {
	    cir.print();
	}
	boundary_config_part.print();
	output_filename_config_part.print();
	external_magnetic_field_config_part.print();
	std::cout << "======" << std::endl;
    }
};

#endif /* _CONFIG_H_ */
