#include "charged_inner_region.h"

bool Charged_inner_region::check_if_particle_inside( Particle &p )
{
    double px = vec3d_x( p.position );
    double py = vec3d_y( p.position );
    double pz = vec3d_z( p.position );
    return check_if_point_inside( px, py, pz );
}

bool Charged_inner_region::check_if_node_inside( Node_reference &node,
						 double dx, double dy, double dz )
{
    return check_if_point_inside( node.x * dx, node.y * dy, node.z * dz );
}

void Charged_inner_region::mark_inner_nodes( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;

    for ( int k = 0; k < nz; k++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int i = 0; i < nx; i++ ) {
		if ( check_if_point_inside( spat_mesh.node_number_to_coordinate_x(i),
					    spat_mesh.node_number_to_coordinate_y(j),
					    spat_mesh.node_number_to_coordinate_z(k) ) ){
		    inner_nodes.emplace_back( i, j, k );
		}
	    }
	}
    }
}

void Charged_inner_region::select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;

    inner_nodes_not_at_domain_edge.reserve( inner_nodes.size() );
    
    for( auto &node : inner_nodes ){
	if( !node.at_domain_edge( nx, ny, nz ) ){
	    inner_nodes_not_at_domain_edge.push_back( node );
	}
    }
}

// Box

Charged_inner_region_box::Charged_inner_region_box(
    Config &conf,
    Charged_inner_region_box_config_part &charged_inner_region_box_conf,
    Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, charged_inner_region_box_conf );
    get_values_from_config( charged_inner_region_box_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
}

void Charged_inner_region_box::check_correctness_of_related_config_fields(
    Config &conf,
    Charged_inner_region_box_config_part &charged_inner_region_box_conf )
{
    // check if region lies inside the domain
}

void Charged_inner_region_box::get_values_from_config(
    Charged_inner_region_box_config_part &charged_inner_region_box_conf )
{
    name = charged_inner_region_box_conf.charged_inner_region_name;
    charge_density = charged_inner_region_box_conf.charged_inner_region_charge_density;
    x_left = charged_inner_region_box_conf.charged_inner_region_box_x_left;
    x_right = charged_inner_region_box_conf.charged_inner_region_box_x_right;
    y_bottom = charged_inner_region_box_conf.charged_inner_region_box_y_bottom;
    y_top = charged_inner_region_box_conf.charged_inner_region_box_y_top;
    z_near = charged_inner_region_box_conf.charged_inner_region_box_z_near;
    z_far = charged_inner_region_box_conf.charged_inner_region_box_z_far;
}


bool Charged_inner_region_box::check_if_point_inside( double x, double y, double z )
{	
    bool in = 
	( x <= x_right ) && ( x >= x_left ) &&
	( y <= y_top ) && ( y >= y_bottom ) &&
	( z <= z_far ) && ( z >= z_near ) ;
    return in;
}
