#ifndef _INNER_REGION_H_
#define _INNER_REGION_H_

#include <iostream>
#include <algorithm>
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
    double x_left;
    double x_right;
    double y_bottom;
    double y_top;
    double z_near;
    double z_far;
    double potential;
public:
    std::vector<Node_reference> inner_nodes;
    std::vector<Node_reference> inner_nodes_not_at_domain_edge;
    std::vector<Node_reference> near_boundary_nodes;
    std::vector<Node_reference> near_boundary_nodes_not_at_domain_edge;
    // possible todo: add_boundary_nodes    
public:
    Inner_region(){};
    Inner_region( Config &conf,
		  Inner_region_config_part &inner_region_conf,
		  Spatial_mesh &spat_mesh );
    // Inner_region( std::string name, double xleft, double xright,
    // 		  double ybottom, double ytop,
    // 		  double znear, double zfar, double phi ) :
    // 	name( name ),
    // 	x_left( xleft ),
    // 	x_right( xright ),
    // 	y_bottom( ybottom ),
    // 	y_top( ytop ),
    // 	z_near( znear ),
    // 	z_far( zfar ),
    // 	potential( phi )
    // 	{};
    virtual ~Inner_region() {};
    void print() {
	std::cout << "Inner region: name = " << name << std::endl;
	std::cout << "x_left = " << x_left << std::endl;
	std::cout << "x_right = " << x_right << std::endl;
	std::cout << "y_bottom = " << y_bottom << std::endl;
	std::cout << "y_top = " << y_top << std::endl;
	std::cout << "z_near = " << z_near << std::endl;
	std::cout << "z_far = " << z_far << std::endl;
	std::cout << "potential = " << potential << std::endl;
    }
    bool check_if_point_inside( double x, double y, double z );
    bool check_if_particle_inside( Particle &p );
    bool check_if_node_inside( Node_reference &node, double dx, double dy, double dz );
    void mark_inner_nodes( Spatial_mesh &spat_mesh );
    void select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void print_inner_nodes() {
	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
	for( auto &node : inner_nodes )
	    node.print();
    };
    void mark_near_boundary_nodes( Spatial_mesh &spat_mesh );
    void select_near_boundary_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void print_near_boundary_nodes() {
	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
	for( auto &node : near_boundary_nodes )
	    node.print();
    };
private:
    void check_correctness_of_related_config_fields( Config &conf, Inner_region_config_part &inner_region_conf );
    void get_values_from_config( Inner_region_config_part &inner_region_conf );
};


class Inner_regions_manager{
public:
    std::vector<Inner_region> regions;
public:
    Inner_regions_manager( Config &conf, Spatial_mesh &spat_mesh )
    {
	for( auto &inner_region_conf : conf.inner_regions_config_part )
	    regions.emplace_back( conf, inner_region_conf, spat_mesh );
    }

    virtual ~Inner_regions_manager() {};    

    bool check_if_particle_inside( Particle &p )
    {
	for( auto &region : regions )
	    region.check_if_particle_inside( p );
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




class Inner_region_with_model
{
public:
    std::string name;
    TopoDS_Shape geometry;
    const double tolerance = 0.001;
    double potential;
public:
    std::vector<Node_reference> inner_nodes;
    std::vector<Node_reference> inner_nodes_not_at_domain_edge;
    std::vector<Node_reference> near_boundary_nodes;
    std::vector<Node_reference> near_boundary_nodes_not_at_domain_edge;
    // possible todo: add_boundary_nodes    
public:
    Inner_region_with_model( Config &conf,
			     Inner_region_with_model_config_part &inner_region_with_model_conf,
			     Spatial_mesh &spat_mesh );
    bool check_if_point_inside( double x, double y, double z );
    bool check_if_particle_inside( Particle &p );
    bool check_if_node_inside( Node_reference &node, double dx, double dy, double dz );
    void mark_inner_nodes( Spatial_mesh &spat_mesh );
    void select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void print_inner_nodes() {
	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
	for( auto &node : inner_nodes )
	    node.print();
    };
    void mark_near_boundary_nodes( Spatial_mesh &spat_mesh );
    void select_near_boundary_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh );
    void print_near_boundary_nodes() {
	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
	for( auto &node : near_boundary_nodes )
	    node.print();
    };
    void print();
    void write_to_file( std::ofstream &output_file );
    virtual ~Inner_region_with_model();
private:
    void check_correctness_of_related_config_fields( Config &conf );
    void get_values_from_config( Inner_region_with_model_config_part &inner_region_with_model_conf );    
    void read_geometry_file( std::string filename );
};



class Inner_regions_with_models_manager{
public:
    std::vector<Inner_region_with_model> regions;
public:
    Inner_regions_with_models_manager( Config &conf, Spatial_mesh &spat_mesh )
    {
	for( auto &inner_region_with_model_conf : conf.inner_regions_with_models_config_part )
	    regions.emplace_back( conf, inner_region_with_model_conf, spat_mesh );
    }

    virtual ~Inner_regions_with_models_manager() {};

    bool check_if_particle_inside( Particle &p )
    {
	for( auto &region : regions )
	    region.check_if_particle_inside( p );
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



// class Inner_region_sphere{
// public:
//     std::string name;
//     double x0;
//     double y0;
//     double z0;
//     double R;
//     double potential;
// public:
//     std::vector<Node_reference> inner_nodes;
//     std::vector<Node_reference> near_boundary_nodes;
// public:
//     Inner_region_sphere(){};
//     Inner_region_sphere( std::string name,
// 			 double x0_arg, double y0_arg, double z0_arg,
// 			 double R_arg,
// 			 double phi ) :
//         name( name ),
// 	x0( x0_arg ),
// 	y0( y0_arg ),
// 	z0( z0_arg ),
// 	R( R_arg ),    
// 	potential( phi )
// 	{};
//     virtual ~Inner_region_sphere() {};
//     void print() {
// 	std::cout << "Inner region: name = " << name << std::endl;
// 	std::cout << "x0 = " << x0 << std::endl;
// 	std::cout << "y0 = " << y0 << std::endl;
// 	std::cout << "z0 = " << z0 << std::endl;
// 	std::cout << "R = " << R << std::endl;
// 	std::cout << "potential = " << potential << std::endl;
//     }
//     bool check_if_point_inside( double x, double y, double z );
//     bool check_if_node_inside( Node_reference &node, double dx, double dy, double dz );
//     void mark_inner_points( double *x, int nx, double *y, int ny, double *z, int nz );
//     std::vector<int> global_indices_of_inner_nodes_not_at_domain_boundary(
// 	int nx, int ny, int nz );
//     void print_inner_points() {
// 	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
// 	for( auto &node : inner_nodes )
// 	    node.print();
//     };
//     void mark_near_boundary_points( double *x, int nx, double *y, int ny, double *z, int nz );
//     void print_near_boundary_points() {
// 	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
// 	for( auto &node : near_boundary_nodes )
// 	    node.print();
//     };
// };

#endif /* _INNER_REGION_H_ */
