#ifndef _PARTICLE_SOURCE_H_
#define _PARTICLE_SOURCE_H_

#include <cmath>
#include <random>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/ptr_container/ptr_vector.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <mpi.h>
#include "config.h"
#include "particle.h"
#include "vec3d.h"

class Particle_source{
public:
    std::string name;
    std::string geometry_type;
    std::vector<Particle> particles;
protected:
    int initial_number_of_particles;
    int particles_to_generate_each_step;
    unsigned int max_id;
    // Momentum
    Vec3d mean_momentum;
    double temperature;
    // Particle characteristics
    double charge;
    double mass;
    // Random number generator
    std::default_random_engine rnd_gen;
public:
    Particle_source( Config &conf, Particle_source_config_part &src_conf );
    Particle_source( hid_t h5_particle_source_group_id );
    void generate_each_step();
    void update_particles_position( double dt );
    void print_particles();
    void write_to_file( hid_t hdf5_file_id );
    virtual ~Particle_source() {};
protected:
    // Initialization
    virtual void set_parameters_from_config( Particle_source_config_part &src_conf );
    virtual void read_hdf5_source_parameters( hid_t h5_particle_source_group_id );
    void read_hdf5_particles( hid_t h5_particle_source_group_id );
    // Particles generation 
    void generate_initial_particles();
    void generate_num_of_particles( int num_of_particles );
    // Todo: replace 'std::default_random_engine' type with something more general.
    virtual Vec3d uniform_position_in_source( std::default_random_engine &rnd_gen ) = 0;
    Vec3d maxwell_momentum_distr( const Vec3d mean_momentum,
				  const double temperature, const double mass,
				  std::default_random_engine &rnd_gen );
    int num_of_particles_for_each_process( int num_of_particles );
    void populate_vec_of_ids( std::vector<int> &vec_of_ids,
			      int num_of_particles_for_this_proc );
    double random_in_range( const double low, const double up,
			    std::default_random_engine &rnd_gen );
    // Check config
    virtual void check_correctness_of_related_config_fields( 
	Config &conf, Particle_source_config_part &src_conf );
    void initial_number_of_particles_gt_zero( 
	Config &conf, Particle_source_config_part &src_conf );
    void particles_to_generate_each_step_ge_zero( 
	Config &conf, Particle_source_config_part &src_conf );
    void temperature_gt_zero( 
	Config &conf, Particle_source_config_part &src_conf );
    void mass_gt_zero( 
	Config &conf, Particle_source_config_part &src_conf );
    // Write to file
    void write_hdf5_particles( hid_t current_source_group_id );
    virtual void write_hdf5_source_parameters( hid_t current_source_group_id );
    void hdf5_status_check( herr_t status );
    int total_particles_across_all_processes();
    int data_offset_for_each_process_for_1d_dataset();
};


class Particle_source_box : public Particle_source {
private:
    // Source position
    double xleft;
    double xright;
    double ytop;
    double ybottom;
    double znear;
    double zfar;
public:
    Particle_source_box( Config &conf, Particle_source_box_config_part &src_conf );
    Particle_source_box( hid_t h5_particle_source_box_group_id );
    virtual ~Particle_source_box() {};
private:
    // Particle generation
    virtual void set_parameters_from_config( Particle_source_box_config_part &src_conf );
    virtual void read_hdf5_source_parameters( hid_t h5_particle_source_box_group_id );
    virtual Vec3d uniform_position_in_source( std::default_random_engine &rnd_gen );
    Vec3d uniform_position_in_cube( const double xleft, const double ytop,
				    const double zfar, const double xright,
				    const double ybottom, const double znear,
				    std::default_random_engine &rnd_gen );
    // Check config
    virtual void check_correctness_of_related_config_fields( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void x_right_ge_zero( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void x_right_le_particle_source_x_left( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void x_left_le_grid_x_size( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void y_bottom_ge_zero( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void y_bottom_le_particle_source_y_top( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void y_top_le_grid_y_size( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void z_near_ge_zero( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void z_near_le_particle_source_z_far( 
	Config &conf, Particle_source_box_config_part &src_conf );
    void z_far_le_grid_z_size( 
	Config &conf, Particle_source_box_config_part &src_conf );
    // Write to file
    virtual void write_hdf5_source_parameters( hid_t current_source_group_id );
};


class Particle_source_cylinder : public Particle_source {
private:
    // Source position
    double axis_start_x;
    double axis_start_y;
    double axis_start_z;
    double axis_end_x;
    double axis_end_y;
    double axis_end_z;
    double radius;
public:
    Particle_source_cylinder( Config &conf,
			      Particle_source_cylinder_config_part &src_conf );
    Particle_source_cylinder( hid_t h5_particle_source_cylinder_group_id );
    virtual ~Particle_source_cylinder() {};
private:
    // Particle generation
    virtual void set_parameters_from_config(
	Particle_source_cylinder_config_part &src_conf );
    virtual void read_hdf5_source_parameters( hid_t h5_particle_source_cylinder_group_id );
    virtual Vec3d uniform_position_in_source( std::default_random_engine &rnd_gen );
    Vec3d uniform_position_in_cylinder( std::default_random_engine &rnd_gen );
    // Check config
    virtual void check_correctness_of_related_config_fields( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void radius_gt_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_x_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_x_plus_rad_le_grid_x_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_y_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_y_plus_rad_le_grid_y_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_z_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_start_z_plus_rad_le_grid_z_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_x_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_x_plus_rad_le_grid_x_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_y_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_y_plus_rad_le_grid_y_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_z_min_rad_ge_zero( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    void axis_end_z_plus_rad_le_grid_z_size( 
	Config &conf, Particle_source_cylinder_config_part &src_conf );
    // Write to file
    virtual void write_hdf5_source_parameters( hid_t current_source_group_id );
};



class Particle_sources_manager{
public:
    boost::ptr_vector<Particle_source> sources;
public:
    Particle_sources_manager( Config &conf )
    {
	for( auto &src_conf : conf.sources_config_part ){
	    if( Particle_source_box_config_part *box_conf =
		dynamic_cast<Particle_source_box_config_part*>( &src_conf ) ){
		sources.push_back( new Particle_source_box( conf,
							    *box_conf ) );
	    } else if( Particle_source_cylinder_config_part *cyl_conf =
	    	       dynamic_cast<Particle_source_cylinder_config_part*>( &src_conf ) ){
	    	sources.push_back( new Particle_source_cylinder( conf,
	    							 *cyl_conf ) );
	    } else {
		std::cout << "In sources_manager constructor: " 
			  << "Unknown config type. Aborting" << std::endl; 
		exit( EXIT_FAILURE );
	    }
	}
    }

    Particle_sources_manager( hid_t h5_particle_sources_group )
    {
	hsize_t nobj;
	ssize_t len;
	herr_t err;
	int otype;
	size_t MAX_NAME = 1024;
	char memb_name_cstr[MAX_NAME];
	hid_t current_src_grpid;
	err = H5Gget_num_objs(h5_particle_sources_group, &nobj);

	for( hsize_t i = 0; i < nobj; i++ ){
	    len = H5Gget_objname_by_idx( h5_particle_sources_group, i, 
					 memb_name_cstr, MAX_NAME );
	    hdf5_status_check( len );
	    otype = H5Gget_objtype_by_idx( h5_particle_sources_group, i );
	    if ( otype == H5G_GROUP ) {
		current_src_grpid = H5Gopen( h5_particle_sources_group,
					     memb_name_cstr, H5P_DEFAULT );
		parse_hdf5_particle_source( current_src_grpid );
		err = H5Gclose( current_src_grpid ); hdf5_status_check( err );
	    }		
	}
    }

    void parse_hdf5_particle_source( hid_t current_src_grpid )
    {
	herr_t status;
	char geometry_type_cstr[50];
	status = H5LTget_attribute_string( current_src_grpid, "./",
					   "geometry_type", geometry_type_cstr );
	hdf5_status_check( status );

	std::string geometry_type( geometry_type_cstr );
	if( geometry_type == "box" ){
	    sources.push_back( new Particle_source_box( current_src_grpid ) );
	} else if ( geometry_type == "cylinder" ) {
	    sources.push_back( new Particle_source_cylinder( current_src_grpid ) );
	} else {
	    std::cout << "In Particle_source_manager constructor-from-h5: "
		      << "Unknown particle_source type. Aborting"
		      << std::endl;
	    exit( EXIT_FAILURE );
	}	
    }
    
    virtual ~Particle_sources_manager() {};

    void write_to_file( hid_t hdf5_file_id )
    {
	hid_t group_id;
	herr_t status;
	int single_element = 1;
	std::string hdf5_groupname = "/Particle_sources";
	int n_of_sources = sources.size();
	group_id = H5Gcreate2( hdf5_file_id, hdf5_groupname.c_str(),
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hdf5_status_check( group_id );

	status = H5LTset_attribute_int( hdf5_file_id,
					hdf5_groupname.c_str(),
					"number_of_sources", &n_of_sources,
					single_element );
	hdf5_status_check( status );
	
	for( auto &src : sources )
	    src.write_to_file( group_id );

	status = H5Gclose( group_id );
	hdf5_status_check( status );
    }; 

    void generate_each_step()
    {
	for( auto &src : sources )
	    src.generate_each_step();
    };

    void print_particles()
    {
	for( auto &src : sources )
	    src.print_particles();
    };

    void update_particles_position( double dt )
    {
	for( auto &src : sources )
	    src.update_particles_position( dt );
    };

    void hdf5_status_check( herr_t status )
    {
	if( status < 0 ){
	    std::cout << "Something went wrong while writing or reading"
		      << "'Particle_sources' group. Aborting."
		      << std::endl;
	    exit( EXIT_FAILURE );
	}
    };
};


#endif /* _PARTICLE_SOURCE_H_ */
