#include "inner_region.h"

Inner_region::Inner_region( Config &conf,
			    Inner_region_config_part &inner_region_conf )
{
    check_correctness_of_related_config_fields( conf, inner_region_conf );
    get_values_from_config( inner_region_conf );
    total_absorbed_particles = 0;
    total_absorbed_charge = 0;
    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
}

void Inner_region::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_config_part &inner_region_conf )
{
    // todo
}

void Inner_region::get_values_from_config( Inner_region_config_part &inner_region_conf )
{
    name = inner_region_conf.name;
    potential = inner_region_conf.potential;
}


Inner_region::Inner_region( hid_t h5_inner_region_group_id )
{
    // Read from h5
    get_values_from_h5( h5_inner_region_group_id );
    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
}

void Inner_region::get_values_from_h5( hid_t h5_inner_region_group_id )
{
    herr_t status;

    size_t grp_name_size = 0;
    char *grp_name = NULL;
    grp_name_size = H5Iget_name( h5_inner_region_group_id, grp_name, grp_name_size );
    grp_name_size = grp_name_size + 1;
    grp_name = new char[ grp_name_size ];
    grp_name_size = H5Iget_name( h5_inner_region_group_id, grp_name, grp_name_size );
    std::string longname = std::string( grp_name );
    name = longname.substr( longname.find_last_of("/") + 1 );
    delete[] grp_name;
    
    status = H5LTget_attribute_double( h5_inner_region_group_id, "./",
				       "potential", &potential );
    hdf5_status_check( status );
    status = H5LTget_attribute_int( h5_inner_region_group_id, "./",
				       "total_absorbed_particles", &total_absorbed_particles );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_group_id, "./",
				       "total_absorbed_charge", &total_absorbed_charge );
    hdf5_status_check( status );
}


bool Inner_region::check_if_particle_inside( Particle &p )
{
    double px = vec3d_x( p.position );
    double py = vec3d_y( p.position );
    double pz = vec3d_z( p.position );
    return check_if_point_inside( px, py, pz );
}

bool Inner_region::check_if_particle_inside_and_count_charge( Particle &p )
{
    bool in_or_out;
    in_or_out = check_if_particle_inside( p );
    if( in_or_out ){
	absorbed_particles_current_timestep_current_proc++;
	absorbed_charge_current_timestep_current_proc += p.charge;
    }
    return in_or_out;
}

bool Inner_region::check_if_node_inside( Node_reference &node,
					 double dx, double dy, double dz )
{
    return check_if_point_inside( node.x * dx, node.y * dy, node.z * dz );
}

void Inner_region::mark_inner_nodes( Spatial_mesh &spat_mesh )
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

void Inner_region::select_inner_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh )
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

void Inner_region::mark_near_boundary_nodes( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;

    // rewrite; 
    for( auto &node : inner_nodes ){
	std::vector<Node_reference> neighbours = node.adjacent_nodes();
	for( auto &nbr : neighbours ){
	    if ( !node.at_domain_edge( nx, ny, nz ) ) {
		if ( !check_if_point_inside( spat_mesh.node_number_to_coordinate_x(nbr.x),
					     spat_mesh.node_number_to_coordinate_y(nbr.y),
					     spat_mesh.node_number_to_coordinate_z(nbr.z) ) ){
		    near_boundary_nodes.emplace_back( nbr.x, nbr.y, nbr.z );
		}
	    }
	}
    }
    std::sort( near_boundary_nodes.begin(), near_boundary_nodes.end() );
    near_boundary_nodes.erase(
	std::unique( near_boundary_nodes.begin(), near_boundary_nodes.end() ),
	near_boundary_nodes.end() );
}

void Inner_region::select_near_boundary_nodes_not_at_domain_edge( Spatial_mesh &spat_mesh )
{
    // todo: repeats with select_inner_nodes_not_at_domain_edge;
    // remove code duplication
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;

    near_boundary_nodes_not_at_domain_edge.reserve( near_boundary_nodes.size() );
    
    for( auto &node : near_boundary_nodes ){
	if( !node.at_domain_edge( nx, ny, nz ) ){
	    near_boundary_nodes_not_at_domain_edge.push_back( node );
	}
    }    
}

void Inner_region::sync_absorbed_charge_and_particles_across_proc()
{
    int single = 1;

    MPI_Allreduce( MPI_IN_PLACE, &absorbed_particles_current_timestep_current_proc, single,
		   MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &absorbed_charge_current_timestep_current_proc, single,
		   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    
    total_absorbed_charge += absorbed_charge_current_timestep_current_proc;
    total_absorbed_particles += absorbed_particles_current_timestep_current_proc;   

    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
}

void Inner_region::write_to_file( hid_t regions_group_id )
{
    hid_t current_region_group_id;
    herr_t status;
    std::string current_region_groupname = name;
    current_region_group_id = H5Gcreate( regions_group_id,
					 current_region_groupname.c_str(),
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_region_group_id );

    write_hdf5_common_parameters( current_region_group_id );
    write_hdf5_region_specific_parameters( current_region_group_id );

    status = H5Gclose( current_region_group_id );
    hdf5_status_check( status );    
}

void Inner_region::write_hdf5_common_parameters( hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";

    status = H5LTset_attribute_string( current_region_group_id,
				       current_region_groupname.c_str(),
				       "object_type", object_type.c_str() );
    hdf5_status_check( status );
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "potential", &potential, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_int( current_region_group_id,
				    current_region_groupname.c_str(),
				    "total_absorbed_particles",
				    &total_absorbed_particles, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "total_absorbed_charge",
				       &total_absorbed_charge, single_element );
    hdf5_status_check( status );
}

void Inner_region::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while writing Inner_region "
		  << name << "."
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}

// Box

Inner_region_box::Inner_region_box(
    Config &conf,
    Inner_region_box_config_part &inner_region_box_conf,
    Spatial_mesh &spat_mesh ) :
    Inner_region( conf, inner_region_box_conf )
{
    object_type = "box";
    check_correctness_of_related_config_fields( conf, inner_region_box_conf );
    get_values_from_config( inner_region_box_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

Inner_region_box::Inner_region_box(
    hid_t h5_inner_region_box_group_id,
    Spatial_mesh &spat_mesh ) :
    Inner_region( h5_inner_region_box_group_id )
{
    object_type = "box";
    //check_correctness_of_related_config_fields( conf, inner_region_box_conf );
    get_values_from_h5( h5_inner_region_box_group_id );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}


void Inner_region_box::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_box_config_part &inner_region_box_conf )
{
    // check if region lies inside the domain
}

void Inner_region_box::get_values_from_config(
    Inner_region_box_config_part &inner_region_box_conf )
{
    x_left = inner_region_box_conf.box_x_left;
    x_right = inner_region_box_conf.box_x_right;
    y_bottom = inner_region_box_conf.box_y_bottom;
    y_top = inner_region_box_conf.box_y_top;
    z_near = inner_region_box_conf.box_z_near;
    z_far = inner_region_box_conf.box_z_far;
}

void Inner_region_box::get_values_from_h5(
        hid_t h5_inner_region_box_group_id )
{
    herr_t status;
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "x_left", &x_left ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "x_right", &x_right ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "y_bottom", &y_bottom ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "y_top", &y_top ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "z_near", &z_near ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_box_group_id, "./",
				       "z_far", &z_far ); hdf5_status_check( status );
}


bool Inner_region_box::check_if_point_inside( double x, double y, double z )
{	
    bool in = 
	( x <= x_left ) && ( x >= x_right ) &&
	( y <= y_top ) && ( y >= y_bottom ) &&
	( z <= z_far ) && ( z >= z_near ) ;
    return in;
}


void Inner_region_box::write_hdf5_region_specific_parameters(
    hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "x_left", &x_left, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "x_right", &x_right, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "y_bottom", &y_bottom, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "y_top", &y_top, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "z_near", &z_near, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "z_far", &z_far, single_element );
    hdf5_status_check( status );
}



// Sphere

Inner_region_sphere::Inner_region_sphere(
    Config &conf,
    Inner_region_sphere_config_part &inner_region_sphere_conf,
    Spatial_mesh &spat_mesh ) :
    Inner_region( conf, inner_region_sphere_conf )
{
    object_type = "sphere";
    check_correctness_of_related_config_fields( conf, inner_region_sphere_conf );
    get_values_from_config( inner_region_sphere_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}


Inner_region_sphere::Inner_region_sphere(
    hid_t h5_inner_region_sphere_group_id,
    Spatial_mesh &spat_mesh ) :
    Inner_region( h5_inner_region_sphere_group_id )
{
    object_type = "sphere";
    //check_correctness_of_related_config_fields( conf, inner_region_sphere_conf );
    get_values_from_h5( h5_inner_region_sphere_group_id );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}


void Inner_region_sphere::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_sphere_config_part &inner_region_sphere_conf )
{
    // check if region lies inside the domain
}

void Inner_region_sphere::get_values_from_config(
    Inner_region_sphere_config_part &inner_region_sphere_conf )
{
    origin_x = inner_region_sphere_conf.sphere_origin_x;
    origin_y = inner_region_sphere_conf.sphere_origin_y;
    origin_z = inner_region_sphere_conf.sphere_origin_z;
    radius = inner_region_sphere_conf.sphere_radius;
}


void Inner_region_sphere::get_values_from_h5(
        hid_t h5_inner_region_sphere_group_id )
{
    herr_t status;
    status = H5LTget_attribute_double( h5_inner_region_sphere_group_id, "./",
				       "origin_x", &origin_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_sphere_group_id, "./",
				       "origin_y", &origin_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_sphere_group_id, "./",
				       "origin_z", &origin_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_sphere_group_id, "./",
				       "radius", &radius ); hdf5_status_check( status );
}



bool Inner_region_sphere::check_if_point_inside( double x, double y, double z )
{
    double xdist = (x - origin_x);
    double ydist = (y - origin_y);
    double zdist = (z - origin_z);
    bool in = ( xdist * xdist + ydist * ydist + zdist * zdist <= radius * radius );
    return in;
}


void Inner_region_sphere::write_hdf5_region_specific_parameters(
	hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "origin_x", &origin_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "origin_y", &origin_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "origin_z", &origin_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "radius", &radius, single_element );
    hdf5_status_check( status );    
}



// Cylinder

Inner_region_cylinder::Inner_region_cylinder(
    Config &conf,
    Inner_region_cylinder_config_part &inner_region_cylinder_conf,
    Spatial_mesh &spat_mesh )
    : Inner_region( conf, inner_region_cylinder_conf )
{
    object_type = "cylinder";
    check_correctness_of_related_config_fields( conf, inner_region_cylinder_conf );
    get_values_from_config( inner_region_cylinder_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

Inner_region_cylinder::Inner_region_cylinder(
    hid_t h5_inner_region_cylinder_group_id,
    Spatial_mesh &spat_mesh ) :
    Inner_region( h5_inner_region_cylinder_group_id )
{
    object_type = "cylinder";
    //check_correctness_of_related_config_fields( conf, inner_region_cylinder_conf );
    get_values_from_h5( h5_inner_region_cylinder_group_id );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}


void Inner_region_cylinder::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_cylinder_config_part &inner_region_cylinder_conf )
{
    // check if region lies inside the domain
}

void Inner_region_cylinder::get_values_from_config(
    Inner_region_cylinder_config_part &inner_region_cylinder_conf )
{
    axis_start_x = inner_region_cylinder_conf.cylinder_axis_start_x;
    axis_start_y = inner_region_cylinder_conf.cylinder_axis_start_y;
    axis_start_z = inner_region_cylinder_conf.cylinder_axis_start_z;
    axis_end_x = inner_region_cylinder_conf.cylinder_axis_end_x;
    axis_end_y = inner_region_cylinder_conf.cylinder_axis_end_y;
    axis_end_z = inner_region_cylinder_conf.cylinder_axis_end_z;
    radius = inner_region_cylinder_conf.cylinder_radius;
}


void Inner_region_cylinder::get_values_from_h5(
        hid_t h5_inner_region_cylinder_group_id )
{
    herr_t status;
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_start_x", &axis_start_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_start_y", &axis_start_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_start_z", &axis_start_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_end_x", &axis_end_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_end_y", &axis_end_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "axis_end_z", &axis_end_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_cylinder_group_id, "./",
				       "radius", &radius ); hdf5_status_check( status );
}



bool Inner_region_cylinder::check_if_point_inside( double x, double y, double z )
{
    Vec3d pointvec = vec3d_init( (x - axis_start_x),
				 (y - axis_start_y),
				 (z - axis_start_z) );
    Vec3d axisvec = vec3d_init( ( axis_end_x - axis_start_x ),
				( axis_end_y - axis_start_y ),
				( axis_end_z - axis_start_z ) );    
    Vec3d unit_axisvec = vec3d_normalized( axisvec );
			   
    double projection = vec3d_dot_product( pointvec, unit_axisvec );
    Vec3d perp_to_axis = vec3d_sub( pointvec,
				    vec3d_times_scalar( unit_axisvec, projection ) );
    bool in = ( projection >= 0 &&
		projection <= vec3d_length( axisvec ) &&
		vec3d_length( perp_to_axis ) <= radius );
    return in;
}


void Inner_region_cylinder::write_hdf5_region_specific_parameters(
    hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_x", &axis_start_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_y", &axis_start_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_z", &axis_start_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_x", &axis_end_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_y", &axis_end_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_z", &axis_end_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "radius", &radius, single_element );
    hdf5_status_check( status );
}



// Tube

Inner_region_tube::Inner_region_tube(
    Config &conf,
    Inner_region_tube_config_part &inner_region_tube_conf,
    Spatial_mesh &spat_mesh )
    : Inner_region( conf, inner_region_tube_conf )
{
    object_type = "tube";
    check_correctness_of_related_config_fields( conf, inner_region_tube_conf );
    get_values_from_config( inner_region_tube_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

Inner_region_tube::Inner_region_tube(
    hid_t h5_inner_region_tube_group_id,
    Spatial_mesh &spat_mesh ) :
    Inner_region( h5_inner_region_tube_group_id )
{
    object_type = "tube";
    //check_correctness_of_related_config_fields( conf, inner_region_tube_conf );
    get_values_from_h5( h5_inner_region_tube_group_id );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

void Inner_region_tube::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_tube_config_part &inner_region_tube_conf )
{
    // check if region lies inside the domain
}

void Inner_region_tube::get_values_from_config(
    Inner_region_tube_config_part &inner_region_tube_conf )
{
    axis_start_x = inner_region_tube_conf.tube_axis_start_x;
    axis_start_y = inner_region_tube_conf.tube_axis_start_y;
    axis_start_z = inner_region_tube_conf.tube_axis_start_z;
    axis_end_x = inner_region_tube_conf.tube_axis_end_x;
    axis_end_y = inner_region_tube_conf.tube_axis_end_y;
    axis_end_z = inner_region_tube_conf.tube_axis_end_z;
    inner_radius = inner_region_tube_conf.tube_inner_radius;
    outer_radius = inner_region_tube_conf.tube_outer_radius;
}

void Inner_region_tube::get_values_from_h5(
        hid_t h5_inner_region_tube_group_id )
{
    herr_t status;
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_start_x", &axis_start_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_start_y", &axis_start_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_start_z", &axis_start_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_end_x", &axis_end_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_end_y", &axis_end_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "axis_end_z", &axis_end_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "inner_radius", &inner_radius ); hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_inner_region_tube_group_id, "./",
				       "outer_radius", &outer_radius ); hdf5_status_check( status );
}



bool Inner_region_tube::check_if_point_inside( double x, double y, double z )
{
    Vec3d pointvec = vec3d_init( (x - axis_start_x),
				 (y - axis_start_y),
				 (z - axis_start_z) );
    Vec3d axisvec = vec3d_init( ( axis_end_x - axis_start_x ),
				( axis_end_y - axis_start_y ),
				( axis_end_z - axis_start_z ) );
    Vec3d unit_axisvec = vec3d_normalized( axisvec );
			   
    double projection = vec3d_dot_product( pointvec, unit_axisvec );
    Vec3d perp_to_axis = vec3d_sub( pointvec,
				    vec3d_times_scalar( unit_axisvec, projection ) );
    bool in = ( projection >= 0 &&
		projection <= vec3d_length( axisvec ) &&
		vec3d_length( perp_to_axis ) >= inner_radius &&
		vec3d_length( perp_to_axis ) <= outer_radius );
    return in;
}


void Inner_region_tube::write_hdf5_region_specific_parameters(
    hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_x", &axis_start_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_y", &axis_start_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_z", &axis_start_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_x", &axis_end_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_y", &axis_end_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_z", &axis_end_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "inner_radius", &inner_radius, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "outer_radius", &outer_radius, single_element );
    hdf5_status_check( status );
}




// Tube_Along_Z_Segment

Inner_region_tube_along_z_segment::Inner_region_tube_along_z_segment(
    Config &conf,
    Inner_region_tube_along_z_segment_config_part &inner_region_tube_along_z_segment_conf,
    Spatial_mesh &spat_mesh )
    : Inner_region( conf, inner_region_tube_along_z_segment_conf )
{
    object_type = "tube_along_z_segment";
    check_correctness_of_related_config_fields(
	conf,
	inner_region_tube_along_z_segment_conf );
    get_values_from_config( inner_region_tube_along_z_segment_conf );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

Inner_region_tube_along_z_segment::Inner_region_tube_along_z_segment(
    hid_t h5_inner_region_tube_along_z_segment_group_id,
    Spatial_mesh &spat_mesh ) :
    Inner_region( h5_inner_region_tube_along_z_segment_group_id )
{
    object_type = "tube_along_z_segment";
    // check_correctness_of_related_config_fields( conf,
    // 						inner_region_tube_along_z_segment_conf );
    get_values_from_h5( h5_inner_region_tube_along_z_segment_group_id );
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

void Inner_region_tube_along_z_segment::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_tube_along_z_segment_config_part &inner_region_tube_along_z_segment_conf )
{
    // check if region lies inside the domain
}

void Inner_region_tube_along_z_segment::get_values_from_config(
    Inner_region_tube_along_z_segment_config_part &inner_region_tube_along_z_segment_conf )
{
    axis_x = inner_region_tube_along_z_segment_conf.tube_along_z_segment_axis_x;
    axis_y = inner_region_tube_along_z_segment_conf.tube_along_z_segment_axis_y;
    axis_start_z =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_axis_start_z;
    axis_end_z =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_axis_end_z;
    inner_radius =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_inner_radius;
    outer_radius =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_outer_radius;
    start_angle_deg =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_start_angle_deg;
    end_angle_deg =
	inner_region_tube_along_z_segment_conf.tube_along_z_segment_end_angle_deg;
}


void Inner_region_tube_along_z_segment::get_values_from_h5(
    hid_t h5_inner_region_tube_along_z_segment_group_id )
{
    herr_t status;
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"axis_x", &axis_x ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"axis_y", &axis_y ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"axis_start_z", &axis_start_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"axis_end_z", &axis_end_z ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"inner_radius", &inner_radius ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"outer_radius", &outer_radius ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"start_angle_deg", &start_angle_deg ); hdf5_status_check( status );
    status = H5LTget_attribute_double(
	h5_inner_region_tube_along_z_segment_group_id, "./",
	"end_angle_deg", &end_angle_deg ); hdf5_status_check( status );
}



bool Inner_region_tube_along_z_segment::check_if_point_inside(
    double x, double y, double z )
{
    bool in_z = axis_start_z <= z && z <= axis_end_z;
    if ( !in_z ){
	return false;
    }
    //
    double shift_x = x - axis_x;
    double shift_y = y - axis_y;
    double p_r = sqrt( shift_x * shift_x + shift_y * shift_y );
    bool in_r = inner_radius <= p_r && p_r <= outer_radius;
    if ( !in_r ){
	return false;
    }
    //
    double p_phi = atan2( shift_y, shift_x ) * 180.0 / M_PI;
    if ( p_phi < 0 ) {
	// [0:0; 90:pi/2; 180-e:pi; 180+e:-pi; 270:-pi/2]
	p_phi = 360 + p_phi;
    }
    bool in_phi = start_angle_deg < p_phi && p_phi < end_angle_deg;
    if ( !in_phi ){
	return false;
    }
    bool in = true;
    return in;
}


void Inner_region_tube_along_z_segment::write_hdf5_region_specific_parameters(
    hid_t current_region_group_id )
{
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = "./";
    
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_x", &axis_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_y", &axis_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_start_z", &axis_start_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "axis_end_z", &axis_end_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "inner_radius", &inner_radius, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "outer_radius", &outer_radius, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "start_angle_deg", &start_angle_deg,
				       single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_region_group_id,
				       current_region_groupname.c_str(),
				       "end_angle_deg", &end_angle_deg,
				       single_element );
    hdf5_status_check( status );
}

