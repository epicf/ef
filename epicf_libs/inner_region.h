#ifndef _INNER_REGION_H_
#define _INNER_REGION_H_


#include <iostream>
#include <algorithm>
#include <boost/ptr_container/ptr_vector.hpp>
#include "config.h"
#include "spatial_mesh.h"
#include "node_reference.h"
#include "particle.h"

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
    double potential;
public:
    std::vector<Node_reference> inner_nodes;
    std::vector<Node_reference> inner_nodes_not_at_domain_edge;
    std::vector<Node_reference> near_boundary_nodes;
    std::vector<Node_reference> near_boundary_nodes_not_at_domain_edge;
    // possible todo: add_boundary_nodes    
public:
    virtual ~Inner_region() {};
    virtual void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z ) = 0;
    virtual bool check_if_particle_inside( Particle &p );
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
};


class Inner_region_cylinder : public Inner_region{
public:
    double origin_x_start;
    double origin_y_start;
    double origin_z_start;
    double origin_x_end;
    double origin_y_end;
    double origin_z_end;
    double radius_in;
    double radius_out;
public:
    Inner_region_cylinder( Config &conf,
			 Inner_region_cylinder_config_part &inner_region_conf,
			 Spatial_mesh &spat_mesh );
    virtual ~Inner_region_cylinder() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "potential = " << potential << std::endl;
	std::cout << "origin_x_start = " << origin_x_start << std::endl;
	std::cout << "origin_y_start = " << origin_y_start << std::endl;
	std::cout << "origin_z_start = " << origin_z_start << std::endl;
	std::cout << "origin_x_end = " << origin_x_end << std::endl;
	std::cout << "origin_y_end = " << origin_y_end << std::endl;
	std::cout << "origin_z_end = " << origin_z_end << std::endl;
	std::cout << "radius_in = " << radius_in << std::endl;
	std::cout << "radius_out = " << radius_out << std::endl;
    }
    virtual bool check_if_point_inside( double x, double y, double z );
private:
    virtual void check_correctness_of_related_config_fields( Config &conf,
							     Inner_region_cylinder_config_part &inner_region_sphere_conf );
    virtual void get_values_from_config( Inner_region_cylinder_config_part &inner_region_sphere_conf );
};


class Inner_region_with_model : public Inner_region
{
public:
    TopoDS_Shape geometry;
    const double tolerance = 0.001;
public:
    Inner_region_with_model( Config &conf,
			     Inner_region_with_model_config_part &inner_region_with_model_conf,
			     Spatial_mesh &spat_mesh );
    virtual bool check_if_point_inside( double x, double y, double z );
    virtual ~Inner_region_with_model();
private:
    void check_correctness_of_related_config_fields( Config &conf,
						     Inner_region_with_model_config_part &inner_region_with_model_conf );
    void get_values_from_config( Inner_region_with_model_config_part &inner_region_with_model_conf );    
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
	    } else if( Inner_region_cylinder_config_part *cylinder_conf =
		dynamic_cast<Inner_region_cylinder_config_part*>( &inner_region_conf ) ){
		regions.push_back( new Inner_region_cylinder( conf,
							    *cylinder_conf,
							    spat_mesh ) );	
	    } else if (	Inner_region_with_model_config_part *with_model_conf =
			dynamic_cast<Inner_region_with_model_config_part*>(
			    &inner_region_conf ) ) {
		regions.push_back( new Inner_region_with_model( conf,
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

};

#endif /* _INNER_REGION_H_ */
