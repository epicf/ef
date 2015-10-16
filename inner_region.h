#ifndef _INNER_REGION_H_
#define _INNER_REGION_H_

#include <iostream>
#include <algorithm>
#include "config.h"
#include "particle.h"

class Node_reference{
public:
    int x, y, z;
public:
    Node_reference( int xx, int yy, int zz ): x(xx), y(yy), z(zz) {};
    virtual ~Node_reference() {};
    std::vector<Node_reference> adjacent_nodes() {
	Node_reference left( x-1, y, z );
	Node_reference right( x+1, y, z );
	Node_reference top( x, y-1, z ); // warning y numeration from top to bottom
	Node_reference bottom( x, y+1, z ); // warning y numeration from top to bottom
	Node_reference near( x, y, z-1 );
	Node_reference far( x, y, z+1 );
	std::vector<Node_reference> neighbours( {left, right, top, bottom, near, far} );
	return neighbours;
    };
    void print() {
	std::cout << "Node = " << x << " " << y << " " << z << std::endl;
    };
    bool at_domain_boundary( int nx, int ny, int nz ){
	return ( x <= 0 || x >= nx - 1 ||
		 y <= 0 || y >= ny - 1 ||
		 z <= 0 || z >= nz - 1 );
    };
    int global_index( int nx, int ny, int nz ){
	// x=1,y=1,z=1 corresponds to eq.no 0
	//return ( (x - 1) + (y - 1) * (nx - 2) + (z - 1) * (nx - 2) * (ny - 2) );

	// warning!! x=1, y=ny-2, z=1 corresponds to eq.no 0
	return ( (x - 1) + ( (ny - 2) - y ) * (nx - 2) + (z - 1) * (nx - 2) * (ny - 2) );
    };
    bool left_from( Node_reference other_node ){
	return x+1 == other_node.x;
    };
    bool right_from( Node_reference other_node ){
	return x-1 == other_node.x;
    };
    bool top_from( Node_reference other_node ){
	// warning y numeration from top to bottom
	return y+1 == other_node.y;
    };
    bool bottom_from( Node_reference other_node ){
	// warning y numeration from top to bottom
	return y-1 == other_node.y;
    };
    bool near_from( Node_reference other_node ){
	return z+1 == other_node.z;
    };
    bool far_from( Node_reference other_node ){
	return z-1 == other_node.z;
    };

    // comparison operators are necessary for std::sort and std::unique to work
    bool operator< ( const Node_reference &other_node ){
	return x < other_node.x ||
	    x == other_node.x && y < other_node.y ||
	    x == other_node.x && y == other_node.y && z < other_node.z;
    };
    bool operator== ( const Node_reference &other_node ){
	return x == other_node.x && y == other_node.y && z == other_node.z;
    };
    
};

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
    std::vector<Node_reference> near_boundary_nodes;
public:
    Inner_region(){};
    Inner_region( Config &conf );
    Inner_region( std::string name, double xleft, double xright,
		  double ybottom, double ytop,
		  double znear, double zfar, double phi ) :
	name( name ),
	x_left( xleft ),
	x_right( xright ),
	y_bottom( ybottom ),
	y_top( ytop ),
	z_near( znear ),
	z_far( zfar ),
	potential( phi )
	{};
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
    void mark_inner_points( double *x, int nx, double *y, int ny, double *z, int nz );
    void print_inner_points() {
	std::cout << "Inner nodes of '" << name << "' object." << std::endl;
	for( auto &node : inner_nodes )
	    node.print();
    };
    std::vector<int> global_indices_of_inner_nodes_not_at_domain_boundary(
	int nx, int ny, int nz );
    void mark_near_boundary_points( double *x, int nx, double *y, int ny, double *z, int nz );
    void print_near_boundary_points() {
	std::cout << "Near-boundary nodes of '" << name << "' object." << std::endl;
	for( auto &node : near_boundary_nodes )
	    node.print();
    };
private:
    void check_correctness_of_related_config_fields( Config &conf );
    void get_values_from_config( Config &conf );
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
