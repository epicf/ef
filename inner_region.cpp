#include "inner_region.h"

Inner_region::Inner_region( Config &conf, Inner_region_config_part &inner_region_conf )
{
    check_correctness_of_related_config_fields( conf, inner_region_conf );
    get_values_from_config( inner_region_conf );
}

void Inner_region::check_correctness_of_related_config_fields( Config &conf,
							       Inner_region_config_part &inner_region_conf )
{
    // check if region lies inside the domain
}

void Inner_region::get_values_from_config( Inner_region_config_part &inner_region_conf )
{
    name = inner_region_conf.inner_region_name;
    x_left = inner_region_conf.inner_region_x_left;
    x_right = inner_region_conf.inner_region_x_right;
    y_bottom = inner_region_conf.inner_region_y_bottom;
    y_top = inner_region_conf.inner_region_y_top;
    z_near = inner_region_conf.inner_region_z_near;
    z_far = inner_region_conf.inner_region_z_far;
    potential = inner_region_conf.inner_region_boundary_potential;
}


bool Inner_region::check_if_point_inside( double x, double y, double z )
{	
    bool in = 
	( x <= x_right ) && ( x >= x_left ) &&
	( y <= y_top ) && ( y >= y_bottom ) &&
	( z <= z_far ) && ( z >= z_near ) ;
    return in;
}

bool Inner_region::check_if_particle_inside( Particle &p )
{
    double px = vec3d_x( p.position );
    double py = vec3d_y( p.position );
    double pz = vec3d_z( p.position );
    return check_if_point_inside( px, py, pz );
}

bool Inner_region::check_if_node_inside( Node_reference &node,
					 double dx, double dy, double dz )
{
    return check_if_point_inside( node.x * dx, node.y * dy, node.z * dz );
}

void Inner_region::mark_inner_points( double *x, int nx,
				      double *y, int ny,
				      double *z, int nz )
{
    for ( int k = 0; k < nz; k++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int i = 0; i < nx; i++ ) {
		if ( check_if_point_inside( x[i], y[j], z[k] ) ){
		    inner_nodes.emplace_back( i, j, k );
		}
	    }
	}
    }
}

std::vector<Node_reference> Inner_region::inner_nodes_not_at_domain_edge( int nx, int ny, int nz )
{
    // todo: rewrite with remove_if or something
    // todo: construct this list once during an object creation
    std::vector<Node_reference> node_list;
    node_list.reserve( inner_nodes.size() );
    
    for( auto &node : inner_nodes ){
	if( !node.at_domain_edge( nx, ny, nz ) ){
	    node_list.push_back( node );
	}
    }
    return node_list;
}


void Inner_region::mark_near_boundary_points( double *x, int nx,
					      double *y, int ny,
					      double *z, int nz )
{
    for( auto &node : inner_nodes ){
	std::vector<Node_reference> neighbours = node.adjacent_nodes();
	for( auto &nbr : neighbours ){
	    if ( !check_if_point_inside( x[nbr.x], y[nbr.y], z[nbr.z] ) ){
		near_boundary_nodes.emplace_back( nbr.x, nbr.y, nbr.z );
	    }
	}
    }
    std::sort( near_boundary_nodes.begin(), near_boundary_nodes.end() );
    near_boundary_nodes.erase(
	std::unique( near_boundary_nodes.begin(), near_boundary_nodes.end() ),
	near_boundary_nodes.end() );
}


// Sphere


// bool Inner_region_sphere::check_if_point_inside( double x, double y, double z )
// {	
//     bool in = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0) <= R*R;
//     return in;
// }

// bool Inner_region_sphere::check_if_node_inside( Node_reference &node, double dx, double dy, double dz )
// {
//     return check_if_point_inside( node.x * dx, node.y * dy, node.z * dz );
// }

// void Inner_region_sphere::mark_inner_points( double *x, int nx, double *y, int ny, double *z, int nz )
// {
//     for ( int k = 0; k < nz; k++ ) {
// 	for ( int j = 0; j < ny; j++ ) {
// 	    for ( int i = 0; i < nx; i++ ) {		
// 		if ( check_if_point_inside( x[i], y[j], z[k] ) ){
// 		    inner_nodes.emplace_back( i, j, k );
// 		}
// 	    }
// 	}
//     }
// }

// std::vector<int> Inner_region_sphere::global_indices_of_inner_nodes_not_at_domain_boundary(
//     int nx, int ny, int nz ){
//     std::vector<int> result;
//     result.reserve( inner_nodes.size() );
//     for( auto &node : inner_nodes ){
// 	if( !node.at_domain_boundary( nx, ny, nz ) ){
// 	    result.push_back( node.global_index( nx, ny, nz ) );
// 	}
//     }
//     return result;
// }


// void Inner_region_sphere::mark_near_boundary_points( double *x, int nx,
// 						     double *y, int ny,
// 						     double *z, int nz )
// {
//     for( auto &node : inner_nodes ){
// 	std::vector<Node_reference> neighbours = node.adjacent_nodes();
// 	for( auto &nbr : neighbours ){
// 	    if ( !check_if_point_inside( x[nbr.x], y[nbr.y], z[nbr.z] ) ){
// 		near_boundary_nodes.emplace_back( nbr.x, nbr.y, nbr.z );
// 	    }
// 	}	
//     }
//     std::sort( near_boundary_nodes.begin(), near_boundary_nodes.end() );
//     near_boundary_nodes.erase(
// 	std::unique( near_boundary_nodes.begin(), near_boundary_nodes.end() ),
// 	near_boundary_nodes.end() );
// }

