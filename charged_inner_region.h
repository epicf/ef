#ifndef _CHARGED_INNER_REGION_H_
#define _CHARGED_INNER_REGION_H_


#include <iostream>
#include <string>
#include <algorithm>
#include <boost/ptr_container/ptr_vector.hpp>
#include "config.h"
#include "spatial_mesh.h"
#include "node_reference.h"
#include "particle.h"



class Charged_inner_region{
public:
    std::string name;
    double charge_density;
public:
    std::vector<Node_reference> inner_nodes;
    std::vector<Node_reference> inner_nodes_not_at_domain_edge;
public:
    virtual ~Charged_inner_region() {};
    virtual void print() {
	std::cout << "Charged inner region: name = " << name << std::endl;
	std::cout << "charge density = " << charge_density << std::endl;
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
private:
    virtual void check_correctness_of_related_config_fields(
	Config &conf,
	Charged_inner_region_config_part &inner_region_conf ){} ;
    virtual void get_values_from_config(
	Charged_inner_region_config_part &inner_region_conf ){};
};


class Charged_inner_region_box : public Charged_inner_region{
public:
    double x_left;
    double x_right;
    double y_bottom;
    double y_top;
    double z_near;
    double z_far;
public:
    Charged_inner_region_box( Config &conf,
			      Charged_inner_region_box_config_part &inner_region_conf,
			      Spatial_mesh &spat_mesh );
    virtual ~Charged_inner_region_box() {};
    void print() {
	std::cout << "Charged inner region: name = " << name << std::endl;
	std::cout << "charge density = " << charge_density << std::endl;
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
	Charged_inner_region_box_config_part &inner_region_box_conf );
    virtual void get_values_from_config(
	Charged_inner_region_box_config_part &inner_region_box_conf );
};


class Charged_inner_regions_manager{
public:
    boost::ptr_vector<Charged_inner_region> regions;
public:
    Charged_inner_regions_manager( Config &conf, Spatial_mesh &spat_mesh )
    {
	for( auto &charged_inner_region_conf : conf.charged_inner_regions_config_part ){
	    if( Charged_inner_region_box_config_part *box_conf =
		dynamic_cast<Charged_inner_region_box_config_part*>( &charged_inner_region_conf ) ){
		regions.push_back( new Charged_inner_region_box( conf,
								 *box_conf,
								 spat_mesh ) );
	    } else {
		std::cout << "In Charged_inner_regions_manager constructor: Unknown config type. Aborting" << std::endl; 
		exit( EXIT_FAILURE );
	    }
	}
    }

    virtual ~Charged_inner_regions_manager() {};

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
};

#endif /* _CHARGED_INNER_REGION_H_ */
