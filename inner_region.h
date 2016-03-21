#ifndef _INNER_REGION_H_
#define _INNER_REGION_H_


#include <iostream>
#include <algorithm>
#include <boost/ptr_container/ptr_vector.hpp>
#include <petscksp.h>
#include "config.h"
#include "spatial_mesh.h"
#include "node_reference.h"
#include "particle.h"
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <string>
#include <oce/STEPControl_Reader.hxx>
#include <oce/TopoDS_Shape.hxx>
#include <oce/BRepTools.hxx>
#include <oce/gp_Pnt.hxx>
#include <oce/BRepPrimAPI_MakeBox.hxx>
#include <oce/BRepClass3d_SolidClassifier.hxx>
#include "config.h"


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
    virtual ~Inner_region() {};
    virtual void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
    }
    virtual void sync_absorbed_charge_and_particles_across_proc();
    virtual bool check_if_point_inside( double x, double y, double z ) = 0;
    virtual bool check_if_particle_inside( Particle &p );
    virtual bool check_if_particle_inside_and_count_charge( Particle &p );
    virtual bool check_if_node_inside( Node_reference &node, double dx, double dy, double dz );
    virtual void mark_inner_nodes( Spatial_mesh &spat_mesh );
    virtual void select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    virtual void print_inner_nodes() {
	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
	for( auto &node : inner_nodes )
	    node.print();
    };
    virtual void mark_near_boundary_nodes( Spatial_mesh &spat_mesh );
    virtual void select_near_boundary_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    virtual void print_near_boundary_nodes() {
	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
	for( auto &node : near_boundary_nodes )
	    node.print();
    };
    // Write to file
    virtual void write_to_file( hid_t regions_group_id );
    void hdf5_status_check( herr_t status );

private:
    virtual void check_correctness_of_related_config_fields( Config &conf,
							     Inner_region_config_part &inner_region_conf ){} ;
    virtual void get_values_from_config( Inner_region_config_part &inner_region_conf ){};
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
    virtual void check_correctness_of_related_config_fields( Config &conf,
							     Inner_region_box_config_part &inner_region_box_conf );
    virtual void get_values_from_config( Inner_region_box_config_part &inner_region_box_conf );
    virtual void write_to_file( hid_t regions_group_id );
};


class Inner_region_sphere : public Inner_region{
public:
    double origin_x;
    double origin_y;
    double origin_z;
    double radius;
public:
    Inner_region_sphere( Config &conf,
			 Inner_region_sphere_config_part &inner_region_conf,
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
    virtual void check_correctness_of_related_config_fields( Config &conf,
							     Inner_region_sphere_config_part &inner_region_sphere_conf );
    virtual void get_values_from_config( Inner_region_sphere_config_part &inner_region_sphere_conf );
    virtual void write_to_file( hid_t regions_group_id );
};



class Inner_region_STEP : public Inner_region
{
public:
    TopoDS_Shape geometry;
    const double tolerance = 0.001;
public:
    Inner_region_STEP( Config &conf,
		       Inner_region_STEP_config_part &inner_region_STEP_conf,
		       Spatial_mesh &spat_mesh );
    virtual bool check_if_point_inside( double x, double y, double z );
    virtual ~Inner_region_STEP();
private:
    void check_correctness_of_related_config_fields( Config &conf,
						     Inner_region_STEP_config_part &inner_region_STEP_conf );
    void get_values_from_config( Inner_region_STEP_config_part &inner_region_STEP_conf );    
    void read_geometry_file( std::string filename );
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
	    } else if (	Inner_region_STEP_config_part *with_model_conf =
			dynamic_cast<Inner_region_STEP_config_part*>(
			    &inner_region_conf ) ) {
		regions.push_back( new Inner_region_STEP( conf,
							  *with_model_conf,
							  spat_mesh ) );
	    } else {
		std::cout << "In Inner_regions_manager constructor: Unknown config type. Aborting" << std::endl; 
		exit( EXIT_FAILURE );
	    }
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
	group_id = H5Gcreate2( hdf5_file_id, hdf5_groupname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hdf5_status_check( group_id );

	status = H5LTset_attribute_int( hdf5_file_id, hdf5_groupname.c_str(),
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
	    std::cout << "Something went wrong while writing Inner_regions group. Aborting."
		      << std::endl;
	    exit( EXIT_FAILURE );
	}
    };
    
};

#endif /* _INNER_REGION_H_ */
