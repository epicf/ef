#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <iostream>
#include <string>
#include <limits>

// Too much similar code.
// Add some macroprogramming or templates.
// Rewrite with inheritance.

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

class Particle_source_config_part{
public:
    std::string name;
    int initial_number_of_particles;
    int particles_to_generate_each_step;
    double mean_momentum_x;
    double mean_momentum_y;
    double mean_momentum_z;
    double temperature;
    double charge;
    double mass;
public:
    Particle_source_config_part(){};
    Particle_source_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	name( name ),
	initial_number_of_particles( ptree.get<int>("initial_number_of_particles") ),
	particles_to_generate_each_step( ptree.get<int>("particles_to_generate_each_step") ),
	mean_momentum_x( ptree.get<double>("mean_momentum_x") ),
	mean_momentum_y( ptree.get<double>("mean_momentum_y") ),
	mean_momentum_z( ptree.get<double>("mean_momentum_z") ),
	temperature( ptree.get<double>("temperature") ),
        charge( ptree.get<double>("charge") ),
	mass( ptree.get<double>("mass") )
	{};
    virtual ~Particle_source_config_part() {};
    virtual void print() { 
	std::cout << "Particle_source: name = " << name << std::endl;
	std::cout << "initial_number_of_particles = " << 
	    initial_number_of_particles << std::endl; 
	std::cout << "particles_to_generate_each_step = " << 
	    particles_to_generate_each_step << std::endl; 
	std::cout << "mean_momentum_x = " << mean_momentum_x << std::endl;
	std::cout << "mean_momentum_y = " << mean_momentum_y << std::endl;
	std::cout << "mean_momentum_z = " << mean_momentum_z << std::endl;
	std::cout << "temperature = " << temperature << std::endl;
	std::cout << "charge = " << charge << std::endl;
	std::cout << "mass = " << mass << std::endl;
    }
};

class Particle_source_box_config_part : public Particle_source_config_part {
public:
    double box_x_left;
    double box_x_right;
    double box_y_bottom;
    double box_y_top;
    double box_z_near;
    double box_z_far;
public:
    Particle_source_box_config_part(){};
    Particle_source_box_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	Particle_source_config_part( name, ptree ),
	box_x_left( ptree.get<double>("box_x_left") ),
	box_x_right( ptree.get<double>("box_x_right") ),
        box_y_bottom( ptree.get<double>("box_y_bottom") ),
	box_y_top( ptree.get<double>("box_y_top") ),
	box_z_near( ptree.get<double>("box_z_near") ),
	box_z_far( ptree.get<double>("box_z_far") )
	{};
    virtual ~Particle_source_box_config_part() {};
    virtual void print() {
	Particle_source_config_part::print();
	std::cout << "box_x_left = " << box_x_left << std::endl;
	std::cout << "box_x_right = " << box_x_right << std::endl;
	std::cout << "box_y_bottom = " << box_y_bottom << std::endl;
	std::cout << "box_y_top = " << box_y_top << std::endl;
	std::cout << "box_z_near = " << box_z_near << std::endl;
	std::cout << "box_z_far = " << box_z_far << std::endl;
    }
};

class Particle_source_cylinder_config_part : public Particle_source_config_part {
public:
    double cylinder_axis_start_x;
    double cylinder_axis_start_y;
    double cylinder_axis_start_z;
    double cylinder_axis_end_x;
    double cylinder_axis_end_y;
    double cylinder_axis_end_z;
    double cylinder_radius;
public:
    Particle_source_cylinder_config_part(){};
    Particle_source_cylinder_config_part( std::string name, boost::property_tree::ptree &ptree ) :
	Particle_source_config_part( name, ptree ),
	cylinder_axis_start_x( ptree.get<double>("cylinder_axis_start_x") ),
	cylinder_axis_start_y( ptree.get<double>("cylinder_axis_start_y") ),
	cylinder_axis_start_z( ptree.get<double>("cylinder_axis_start_z") ),
	cylinder_axis_end_x( ptree.get<double>("cylinder_axis_end_x") ),
	cylinder_axis_end_y( ptree.get<double>("cylinder_axis_end_y") ),
	cylinder_axis_end_z( ptree.get<double>("cylinder_axis_end_z") ),
	cylinder_radius( ptree.get<double>("cylinder_radius") )
	{};
    virtual ~Particle_source_cylinder_config_part() {};
    virtual void print() {
	Particle_source_config_part::print();
	std::cout << "cylinder_axis_start_x = " << cylinder_axis_start_x << std::endl;
	std::cout << "cylinder_axis_start_y = " << cylinder_axis_start_y << std::endl;
	std::cout << "cylinder_axis_start_z = " << cylinder_axis_start_z << std::endl;
	std::cout << "cylinder_axis_end_x = " << cylinder_axis_end_x << std::endl;
	std::cout << "cylinder_axis_end_y = " << cylinder_axis_end_y << std::endl;
	std::cout << "cylinder_axis_end_z = " << cylinder_axis_end_z << std::endl;
	std::cout << "cylinder_radius = " << cylinder_radius << std::endl;
    }
};



class Inner_region_config_part {
public:
    std::string name;
    double potential;
public:
    Inner_region_config_part(){};
    Inner_region_config_part(
	std::string name, boost::property_tree::ptree &ptree ):
	name( name ),
	potential( ptree.get<double>("potential") )
	{};	
    virtual ~Inner_region_config_part() {};
    virtual void print() {
	std::cout << "Inner region:" << std::endl;
	std::cout << "name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
    }
};

class Inner_region_box_config_part : public Inner_region_config_part{
public:
    double box_x_left;
    double box_x_right;
    double box_y_bottom;
    double box_y_top;
    double box_z_near;
    double box_z_far;
public:
    Inner_region_box_config_part(){};
    Inner_region_box_config_part(
	std::string name, boost::property_tree::ptree &ptree ) :
	Inner_region_config_part( name, ptree ),
	box_x_left( ptree.get<double>("box_x_left") ),
	box_x_right( ptree.get<double>("box_x_right") ),
        box_y_bottom( ptree.get<double>("box_y_bottom") ),
	box_y_top( ptree.get<double>("box_y_top") ),
	box_z_near( ptree.get<double>("box_z_near") ),
	box_z_far( ptree.get<double>("box_z_far") )
	{};
    virtual ~Inner_region_box_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "box_x_left = " << box_x_left << std::endl;
	std::cout << "box_x_right = " << box_x_right << std::endl;
	std::cout << "box_y_bottom = " << box_y_bottom << std::endl;
	std::cout << "box_y_top = " << box_y_top << std::endl;
	std::cout << "box_z_near = " << box_z_near << std::endl;
	std::cout << "box_z_far = " << box_z_far << std::endl;
    }
};

class Inner_region_sphere_config_part : public Inner_region_config_part{
public:
    double sphere_origin_x;
    double sphere_origin_y;
    double sphere_origin_z;
    double sphere_radius;
public:
    Inner_region_sphere_config_part(){};
    Inner_region_sphere_config_part(
	std::string name, boost::property_tree::ptree &ptree ) :
	Inner_region_config_part( name, ptree ),
	sphere_origin_x( ptree.get<double>("sphere_origin_x") ),
	sphere_origin_y( ptree.get<double>("sphere_origin_y") ),
	sphere_origin_z( ptree.get<double>("sphere_origin_z") ),
	sphere_radius( ptree.get<double>("sphere_radius") )
	{};
    virtual ~Inner_region_sphere_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "sphere_origin_x = " << sphere_origin_x << std::endl;
	std::cout << "sphere_origin_y = " << sphere_origin_y << std::endl;
	std::cout << "sphere_origin_z = " << sphere_origin_z << std::endl;
	std::cout << "sphere_radius = " << sphere_radius << std::endl;
    }
};


class Inner_region_cylinder_config_part : public Inner_region_config_part{
public:
    double cylinder_axis_start_x;
    double cylinder_axis_start_y;
    double cylinder_axis_start_z;
    double cylinder_axis_end_x;
    double cylinder_axis_end_y;
    double cylinder_axis_end_z;
    double cylinder_radius;
public:
    Inner_region_cylinder_config_part(){};
    Inner_region_cylinder_config_part(
	std::string name, boost::property_tree::ptree &ptree ) :
	Inner_region_config_part( name, ptree ),
	cylinder_axis_start_x( ptree.get<double>("cylinder_axis_start_x") ),
	cylinder_axis_start_y( ptree.get<double>("cylinder_axis_start_y") ),
	cylinder_axis_start_z( ptree.get<double>("cylinder_axis_start_z") ),
	cylinder_axis_end_x( ptree.get<double>("cylinder_axis_end_x") ),
	cylinder_axis_end_y( ptree.get<double>("cylinder_axis_end_y") ),
	cylinder_axis_end_z( ptree.get<double>("cylinder_axis_end_z") ),
	cylinder_radius( ptree.get<double>("cylinder_radius") )
	{};
    virtual ~Inner_region_cylinder_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "inner_region_potential = " << potential << std::endl;
	std::cout << "cylinder_axis_start_x = " << cylinder_axis_start_x << std::endl;
	std::cout << "cylinder_axis_start_y = " << cylinder_axis_start_y << std::endl;
	std::cout << "cylinder_axis_start_z = " << cylinder_axis_start_z << std::endl;
	std::cout << "cylinder_axis_end_x = " << cylinder_axis_end_x << std::endl;
	std::cout << "cylinder_axis_end_y = " << cylinder_axis_end_y << std::endl;
	std::cout << "cylinder_axis_end_z = " << cylinder_axis_end_z << std::endl;
	std::cout << "cylinder_radius = " << cylinder_radius << std::endl;
    }
};


class Inner_region_tube_config_part : public Inner_region_config_part{
public:
    double tube_axis_start_x;
    double tube_axis_start_y;
    double tube_axis_start_z;
    double tube_axis_end_x;
    double tube_axis_end_y;
    double tube_axis_end_z;
    double tube_inner_radius;
    double tube_outer_radius;
public:
    Inner_region_tube_config_part(){};
    Inner_region_tube_config_part(
	std::string name, boost::property_tree::ptree &ptree ) :
	Inner_region_config_part( name, ptree ),
	tube_axis_start_x( ptree.get<double>("tube_axis_start_x") ),
	tube_axis_start_y( ptree.get<double>("tube_axis_start_y") ),
	tube_axis_start_z( ptree.get<double>("tube_axis_start_z") ),
	tube_axis_end_x( ptree.get<double>("tube_axis_end_x") ),
	tube_axis_end_y( ptree.get<double>("tube_axis_end_y") ),
	tube_axis_end_z( ptree.get<double>("tube_axis_end_z") ),
	tube_inner_radius( ptree.get<double>("tube_inner_radius") ),
	tube_outer_radius( ptree.get<double>("tube_outer_radius") )
	{};
    virtual ~Inner_region_tube_config_part() {};
    void print() { 
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "tube_axis_start_x = " << tube_axis_start_x << std::endl;
	std::cout << "tube_axis_start_y = " << tube_axis_start_y << std::endl;
	std::cout << "tube_axis_start_z = " << tube_axis_start_z << std::endl;
	std::cout << "tube_axis_end_x = " << tube_axis_end_x << std::endl;
	std::cout << "tube_axis_end_y = " << tube_axis_end_y << std::endl;
	std::cout << "tube_axis_end_z = " << tube_axis_end_z << std::endl;
	std::cout << "tube_inner_radius = " << tube_inner_radius << std::endl;
	std::cout << "tube_outer_radius = " << tube_outer_radius << std::endl;
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


class Particle_interaction_model_config_part {
public:
    std::string particle_interaction_model;
public:
    Particle_interaction_model_config_part(){};
    Particle_interaction_model_config_part( boost::property_tree::ptree &ptree ) :
	particle_interaction_model( ptree.get<std::string>("particle_interaction_model") )
	{} ;
    virtual ~Particle_interaction_model_config_part() {};
    void print() {
	std::cout << "Particle_interaction_model = " << particle_interaction_model << std::endl;
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
    boost::ptr_vector<Particle_source_config_part> sources_config_part;
    boost::ptr_vector<Inner_region_config_part> inner_regions_config_part;
    Boundary_config_part boundary_config_part;
    External_magnetic_field_config_part external_magnetic_field_config_part;
    Particle_interaction_model_config_part particle_interaction_model_config_part;
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
	particle_interaction_model_config_part.print();
	output_filename_config_part.print();
	external_magnetic_field_config_part.print();
	std::cout << "======" << std::endl;
    }
};

#endif /* _CONFIG_H_ */
