#include "inner_region.h"

Inner_region::Inner_region( Config &conf )
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );    
}

void Inner_region::check_correctness_of_related_config_fields( Config &conf )
{
    // check if region lies inside the domain
}

void Inner_region::get_values_from_config( Config &conf )
{
    name = conf.inner_regions_config_part[0].inner_region_name;
    x_left = conf.inner_regions_config_part[0].inner_region_x_left;
    x_right = conf.inner_regions_config_part[0].inner_region_x_right;
    y_bottom = conf.inner_regions_config_part[0].inner_region_y_bottom;
    y_top = conf.inner_regions_config_part[0].inner_region_y_top;
    z_near = conf.inner_regions_config_part[0].inner_region_z_near;
    z_far = conf.inner_regions_config_part[0].inner_region_z_far;
    potential = conf.inner_regions_config_part[0].inner_region_boundary_potential;
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

std::vector<int> Inner_region::global_indices_of_inner_nodes_not_at_domain_boundary(
    int nx, int ny, int nz ){
    std::vector<int> result;
    result.reserve( inner_nodes.size() );
    for( auto &node : inner_nodes ){
	if( !node.at_domain_boundary( nx, ny, nz ) ){
	    result.push_back( node.global_index( nx, ny, nz ) );
	}
    }
    return result;
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

