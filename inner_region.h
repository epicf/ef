#ifndef _INNER_REGION_H_
#define _INNER_REGION_H_

#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <boost/ptr_container/ptr_vector.hpp>
#include <petscksp.h>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "config.h"
#include "spatial_mesh.h"
#include "node_reference.h"
#include "particle.h"
#include "vec3d.h"

class Inner_region{
public:
    std::string name;
    std::string object_type;
    double potential;
    int total_absorbed_particles;
    double total_absorbed_charge;
    int absorbed_particles_current_timestep_current_proc;
    double absorbed_charge_current_timestep_current_proc;
public:
    std::vector<Node_reference> inner_nodes;
    std::vector<Node_reference> inner_nodes_not_at_domain_edge;
    std::vector<Node_reference> near_boundary_nodes;
    std::vector<Node_reference> near_boundary_nodes_not_at_domain_edge;
    // possible todo: add_boundary_nodes
    // Approx solution and RHS inside region;
    // Should be used in MatZeroRows call, but it seems, it has no effect
    Vec phi_inside_region, rhs_inside_region;
public:
    Inner_region(
	Config &conf,
	Inner_region_config_part &inner_region_conf );
    Inner_region(
	hid_t h5_inner_region_group_id );
    virtual ~Inner_region() {};
    virtual void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
    }
    void sync_absorbed_charge_and_particles_across_proc();
    virtual bool check_if_point_inside( double x, double y, double z ) = 0;
    bool check_if_particle_inside( Particle &p );
    bool check_if_particle_inside_and_count_charge( Particle &p );
    bool check_if_node_inside( Node_reference &node, double dx, double dy, double dz );
    void print_inner_nodes() {
	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
	for( auto &node : inner_nodes )
	    node.print();
    };
    void print_near_boundary_nodes() {
	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
	for( auto &node : near_boundary_nodes )
	    node.print();
    };
    // Write to file
    void write_to_file( hid_t regions_group_id );
    void hdf5_status_check( herr_t status );
protected:
    void mark_inner_nodes( Spatial_mesh &spat_mesh );
    void select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void mark_near_boundary_nodes( Spatial_mesh &spat_mesh );
    void select_near_boundary_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void write_hdf5_common_parameters( hid_t current_region_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id ) = 0;
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_config_part &inner_region_conf );
    virtual void get_values_from_config(
	Inner_region_config_part &inner_region_conf );
    virtual void get_values_from_h5(
	hid_t h5_inner_region_group_id );
};


class Inner_region_box : public Inner_region{
public:
    double x_left;
    double x_right;
    double y_bottom;
    double y_top;
    double z_near;
    double z_far;
public:
    Inner_region_box( Config &conf,
		      Inner_region_box_config_part &inner_region_conf,
		      Spatial_mesh &spat_mesh );
    Inner_region_box( hid_t h5_inner_region_box_group_id,
		      Spatial_mesh &spat_mesh );
    virtual ~Inner_region_box() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "x_left = " << x_left << std::endl;
	std::cout << "x_right = " << x_right << std::endl;
	std::cout << "y_bottom = " << y_bottom << std::endl;
	std::cout << "y_top = " << y_top << std::endl;
	std::cout << "z_near = " << z_near << std::endl;
	std::cout << "z_far = " << z_far << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_box_config_part &inner_region_box_conf );
    virtual void get_values_from_config(
	Inner_region_box_config_part &inner_region_box_conf );
    virtual void get_values_from_h5(
        hid_t h5_inner_region_box_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id );
};


class Inner_region_sphere : public Inner_region{
public:
    double origin_x;
    double origin_y;
    double origin_z;
    double radius;
public:
    Inner_region_sphere(
	Config &conf,
	Inner_region_sphere_config_part &inner_region_conf,
	Spatial_mesh &spat_mesh );
    Inner_region_sphere( hid_t h5_inner_region_group_id,
			 Spatial_mesh &spat_mesh );
    virtual ~Inner_region_sphere() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "origin_x = " << origin_x << std::endl;
	std::cout << "origin_y = " << origin_y << std::endl;
	std::cout << "origin_z = " << origin_z << std::endl;
	std::cout << "radius = " << radius << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_sphere_config_part &inner_region_sphere_conf );
    virtual void get_values_from_config(
	Inner_region_sphere_config_part &inner_region_sphere_conf );
    virtual void get_values_from_h5(
        hid_t h5_inner_region_sphere_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id );
};


class Inner_region_cylinder : public Inner_region{
public:
    double axis_start_x;
    double axis_start_y;
    double axis_start_z;
    double axis_end_x;
    double axis_end_y;
    double axis_end_z;
    double radius;
public:
    Inner_region_cylinder(
	Config &conf,
	Inner_region_cylinder_config_part &inner_region_conf,
	Spatial_mesh &spat_mesh );
    Inner_region_cylinder( hid_t h5_inner_region_group_id,
			   Spatial_mesh &spat_mesh );
    virtual ~Inner_region_cylinder() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "axis_start_x = " << axis_start_x << std::endl;
	std::cout << "axis_start_y = " << axis_start_y << std::endl;
	std::cout << "axis_start_z = " << axis_start_z << std::endl;
	std::cout << "axis_end_x = " << axis_end_x << std::endl;
	std::cout << "axis_end_y = " << axis_end_y << std::endl;
	std::cout << "axis_end_z = " << axis_end_z << std::endl;
	std::cout << "radius = " << radius << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );

private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_cylinder_config_part &inner_region_cylinder_conf );
    virtual void get_values_from_config(
	Inner_region_cylinder_config_part &inner_region_cylinder_conf );
    virtual void get_values_from_h5(
        hid_t h5_inner_region_cylinder_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id );
};


class Inner_region_tube : public Inner_region{
public:
    double axis_start_x;
    double axis_start_y;
    double axis_start_z;
    double axis_end_x;
    double axis_end_y;
    double axis_end_z;
    double inner_radius;
    double outer_radius;
public:
    Inner_region_tube(
	Config &conf,
	Inner_region_tube_config_part &inner_region_conf,
	Spatial_mesh &spat_mesh );
    Inner_region_tube( hid_t h5_inner_region_group_id,
		       Spatial_mesh &spat_mesh );
    virtual ~Inner_region_tube() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "axis_start_x = " << axis_start_x << std::endl;
	std::cout << "axis_start_y = " << axis_start_y << std::endl;
	std::cout << "axis_start_z = " << axis_start_z << std::endl;
	std::cout << "axis_end_x = " << axis_end_x << std::endl;
	std::cout << "axis_end_y = " << axis_end_y << std::endl;
	std::cout << "axis_end_z = " << axis_end_z << std::endl;
	std::cout << "inner_radius = " << inner_radius << std::endl;
	std::cout << "outer_radius = " << outer_radius << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_tube_config_part &inner_region_tube_conf );
    virtual void get_values_from_config(
	Inner_region_tube_config_part &inner_region_tube_conf );
    virtual void get_values_from_h5(
        hid_t h5_inner_region_tube_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id );
};


class Inner_region_tube_along_z_segment : public Inner_region{
public:
    double axis_x;
    double axis_y;
    double axis_start_z;
    double axis_end_z;
    double inner_radius;
    double outer_radius;
    double start_angle_deg;
    double end_angle_deg;
public:
    Inner_region_tube_along_z_segment(
	Config &conf,
	Inner_region_tube_along_z_segment_config_part &inner_region_conf,
	Spatial_mesh &spat_mesh );
    Inner_region_tube_along_z_segment( hid_t h5_inner_region_group_id,
				       Spatial_mesh &spat_mesh );
    virtual ~Inner_region_tube_along_z_segment() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "axis_x = " << axis_x << std::endl;
	std::cout << "axis_y = " << axis_y << std::endl;
	std::cout << "axis_start_z = " << axis_start_z << std::endl;
	std::cout << "axis_end_z = " << axis_end_z << std::endl;
	std::cout << "inner_radius = " << inner_radius << std::endl;
	std::cout << "outer_radius = " << outer_radius << std::endl;
	std::cout << "start_angle_deg = " << start_angle_deg << std::endl;
	std::cout << "end_angle_deg = " << end_angle_deg << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Inner_region_tube_along_z_segment_config_part
	&inner_region_tube_along_z_segment_conf );
    virtual void get_values_from_config(
	Inner_region_tube_along_z_segment_config_part
	&inner_region_tube_along_z_segment_conf );
    virtual void get_values_from_h5(
        hid_t h5_inner_region_tube_along_z_segment_group_id );
    virtual void write_hdf5_region_specific_parameters(
	hid_t current_region_group_id );
};



class Inner_regions_manager{
public:
    boost::ptr_vector<Inner_region> regions;
public:
    Inner_regions_manager( Config &conf, Spatial_mesh &spat_mesh )
    {
	for( auto &inner_region_conf : conf.inner_regions_config_part ){
	    if( Inner_region_box_config_part *box_conf =
		dynamic_cast<Inner_region_box_config_part*>( &inner_region_conf ) ){
		regions.push_back( new Inner_region_box( conf,
							 *box_conf,
							 spat_mesh ) );
	    } else if( Inner_region_sphere_config_part *sphere_conf =
		dynamic_cast<Inner_region_sphere_config_part*>( &inner_region_conf ) ){
		regions.push_back( new Inner_region_sphere( conf,
							    *sphere_conf,
							    spat_mesh ) );
	    } else if( Inner_region_cylinder_config_part *cyl_conf =
		dynamic_cast<Inner_region_cylinder_config_part*>( &inner_region_conf ) ){
		regions.push_back( new Inner_region_cylinder( conf,
							      *cyl_conf,
							      spat_mesh ) );
	    } else if( Inner_region_tube_config_part *tube_conf =
		dynamic_cast<Inner_region_tube_config_part*>( &inner_region_conf ) ){
		regions.push_back( new Inner_region_tube( conf,
							  *tube_conf,
							  spat_mesh ) );
	    } else if( Inner_region_tube_along_z_segment_config_part
		       *tube_along_z_segment_conf =
		       dynamic_cast<Inner_region_tube_along_z_segment_config_part*>(
			   &inner_region_conf ) ){
		regions.push_back( new Inner_region_tube_along_z_segment(
				       conf,
				       *tube_along_z_segment_conf,
				       spat_mesh ) );
	    } else {
		std::cout << "In Inner_regions_manager constructor-from-conf: "
			  << "Unknown inner_region type. Aborting"
			  << std::endl;
		exit( EXIT_FAILURE );
	    }
	}
    }

    Inner_regions_manager( hid_t h5_inner_region_group, Spatial_mesh &spat_mesh )
    {	
	hsize_t nobj;
	ssize_t len;
	herr_t err;
	int otype;
	size_t MAX_NAME = 1024;
	char memb_name_cstr[MAX_NAME];
	hid_t current_ir_grpid;
	err = H5Gget_num_objs(h5_inner_region_group, &nobj);

	for( hsize_t i = 0; i < nobj; i++ ){
	    len = H5Gget_objname_by_idx(h5_inner_region_group, i, 
					memb_name_cstr, MAX_NAME );
	    hdf5_status_check( len );
	    otype = H5Gget_objtype_by_idx( h5_inner_region_group, i );
	    if ( otype == H5G_GROUP ) {
		current_ir_grpid = H5Gopen( h5_inner_region_group,
					    memb_name_cstr, H5P_DEFAULT );
		parse_hdf5_inner_reg( current_ir_grpid, spat_mesh );
		err = H5Gclose( current_ir_grpid ); hdf5_status_check( err );
	    }		
	}
	
	// To iterate over subgroups, there is H5Giterate function.
	// However, it needs a callback function as one of it's arguments.
	// H5Giterate( h5_inner_region_group, "./", NULL,
	//               parse_hdf5_inner_reg, &spat_mesh );	
	// It is difficult to pass a C++ method as a callback.
	// See: https://isocpp.org/wiki/faq/pointers-to-members#memfnptr-vs-fnptr
	// Therefore, iteration is manual.
    }
    
    void parse_hdf5_inner_reg( hid_t current_ir_grpid, Spatial_mesh &spat_mesh )
    {
	herr_t status;
	char object_type_cstr[50];
	status = H5LTget_attribute_string( current_ir_grpid, "./",
					   "object_type", object_type_cstr );
	hdf5_status_check( status );

	std::string obj_type( object_type_cstr );
	if( obj_type == "box" ){
	    regions.push_back( new Inner_region_box( current_ir_grpid,
						     spat_mesh ) );
	} else if ( obj_type == "sphere" ) {
	    regions.push_back( new Inner_region_sphere( current_ir_grpid,
							spat_mesh ) );
	} else if ( obj_type == "cylinder" ) {
	    regions.push_back( new Inner_region_cylinder( current_ir_grpid,
							  spat_mesh ) );
	} else if ( obj_type == "tube" ) {
	    regions.push_back( new Inner_region_tube( current_ir_grpid,
						      spat_mesh ) );
	} else if ( obj_type == "tube_along_z_segment" ) {
	    regions.push_back( new Inner_region_tube_along_z_segment( current_ir_grpid,
								      spat_mesh ) );
	} else {
	    std::cout << "In Inner_regions_manager constructor-from-h5: "
		      << "Unknown inner_region type. Aborting"
		      << std::endl;
	    exit( EXIT_FAILURE );
	}	
    }

    virtual ~Inner_regions_manager() {};    

    bool check_if_particle_inside( Particle &p )
    {
	for( auto &region : regions ){
	    if( region.check_if_particle_inside( p ) )
		return true;
	}
	return false;
    }

    bool check_if_particle_inside_and_count_charge( Particle &p )
    {
	for( auto &region : regions ){
	    if( region.check_if_particle_inside_and_count_charge( p ) )
		return true;
	}
	return false;
    }

    void sync_absorbed_charge_and_particles_across_proc()
    {
	for( auto &region : regions )
	    region.sync_absorbed_charge_and_particles_across_proc();
    }
    
    void print( )
    {
	for( auto &region : regions )
	    region.print();
    }

    void print_inner_nodes() {
    	for( auto &region : regions )
	    region.print_inner_nodes();
    }

    void print_near_boundary_nodes() {
    	for( auto &region : regions )
	    region.print_near_boundary_nodes();
    }

    void write_to_file( hid_t hdf5_file_id )
    {
	hid_t group_id;
	herr_t status;
	int single_element = 1;
	std::string hdf5_groupname = "/Inner_regions";
	int n_of_regions = regions.size();
	group_id = H5Gcreate2(
	    hdf5_file_id, hdf5_groupname.c_str(),
	    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
	hdf5_status_check( group_id );

	status = H5LTset_attribute_int(
	    hdf5_file_id, hdf5_groupname.c_str(),
	    "number_of_regions", &n_of_regions, single_element );
	hdf5_status_check( status );
	
	for( auto &reg : regions )
	    reg.write_to_file( group_id );

	status = H5Gclose(group_id);
	hdf5_status_check( status );
    }; 

    void hdf5_status_check( herr_t status )
    {
	if( status < 0 ){
	    std::cout << "Something went wrong while "
		      << "writing or reading Inner_regions group. Aborting."
		      << std::endl;
	    exit( EXIT_FAILURE );
	}
    };
    
};

#endif /* _INNER_REGION_H_ */
