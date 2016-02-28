#include "inner_region.h"

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
    int single_element = 1;
    std::string current_region_groupname = name;
    current_region_group_id = H5Gcreate( regions_group_id, current_region_groupname.c_str(),
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_region_group_id );

    status = H5LTset_attribute_string( regions_group_id, current_region_groupname.c_str(),
				       "object_type", object_type.c_str() );
    
    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "potential", &potential, single_element ); hdf5_status_check( status );

    status = H5LTset_attribute_int( regions_group_id, current_region_groupname.c_str(),
				    "total_absorbed_particles", &total_absorbed_particles, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "total_absorbed_charge", &total_absorbed_charge, single_element );
    hdf5_status_check( status );
	
    status = H5Gclose(current_region_group_id); hdf5_status_check( status );
    return;

    
    // todo: separate functions
    // write_hdf5_absorbed_charge( group_id, region_name );
    
    // todo: write rest of region parameters.
    // this is region-type specific. 
    // write_hdf5_region_parameters( group_id, region_name );
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

Inner_region_box::Inner_region_box( Config &conf,
				    Inner_region_box_config_part &inner_region_box_conf,
				    Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_box_conf );
    get_values_from_config( inner_region_box_conf );
    object_type = "box";
    total_absorbed_particles = 0;
    total_absorbed_charge = 0;
    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
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

void Inner_region_box::get_values_from_config( Inner_region_box_config_part &inner_region_box_conf )
{
    name = inner_region_box_conf.inner_region_name;
    potential = inner_region_box_conf.inner_region_potential;
    x_left = inner_region_box_conf.inner_region_box_x_left;
    x_right = inner_region_box_conf.inner_region_box_x_right;
    y_bottom = inner_region_box_conf.inner_region_box_y_bottom;
    y_top = inner_region_box_conf.inner_region_box_y_top;
    z_near = inner_region_box_conf.inner_region_box_z_near;
    z_far = inner_region_box_conf.inner_region_box_z_far;
}


bool Inner_region_box::check_if_point_inside( double x, double y, double z )
{	
    bool in = 
	( x <= x_right ) && ( x >= x_left ) &&
	( y <= y_top ) && ( y >= y_bottom ) &&
	( z <= z_far ) && ( z >= z_near ) ;
    return in;
}


void Inner_region_box::write_to_file( hid_t regions_group_id )
{
    hid_t current_region_group_id;
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = name;
    
    current_region_group_id = H5Gcreate( regions_group_id, current_region_groupname.c_str(),
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_region_group_id );

    status = H5LTset_attribute_string( regions_group_id, current_region_groupname.c_str(),
				       "object_type", object_type.c_str() );
    
    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "potential", &potential, single_element ); hdf5_status_check( status );

    status = H5LTset_attribute_int( regions_group_id, current_region_groupname.c_str(),
				    "total_absorbed_particles", &total_absorbed_particles, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "total_absorbed_charge", &total_absorbed_charge, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "x_left", &x_left, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "x_right", &x_right, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "y_bottom", &y_bottom, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "y_top", &y_top, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "z_near", &z_near, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "z_far", &z_far, single_element );
    hdf5_status_check( status );
    
    status = H5Gclose(current_region_group_id); hdf5_status_check( status );
    return;
    
    // todo: separate functions
    // write_hdf5_absorbed_charge( group_id, region_name );

    // todo: call Inner_region::write_to_file()
    // to write common properties.
    // then write rest of parameters
    // write_hdf5_region_parameters( group_id, region_name );
}



// Sphere

Inner_region_sphere::Inner_region_sphere( Config &conf,
					  Inner_region_sphere_config_part &inner_region_sphere_conf,
					  Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_sphere_conf );
    get_values_from_config( inner_region_sphere_conf );
    object_type = "sphere";
    total_absorbed_particles = 0;
    total_absorbed_charge = 0;
    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
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

void Inner_region_sphere::get_values_from_config( Inner_region_sphere_config_part &inner_region_sphere_conf )
{
    name = inner_region_sphere_conf.inner_region_name;
    potential = inner_region_sphere_conf.inner_region_potential;
    origin_x = inner_region_sphere_conf.inner_region_sphere_origin_x;
    origin_y = inner_region_sphere_conf.inner_region_sphere_origin_y;
    origin_z = inner_region_sphere_conf.inner_region_sphere_origin_z;
    radius = inner_region_sphere_conf.inner_region_sphere_radius;
}


bool Inner_region_sphere::check_if_point_inside( double x, double y, double z )
{
    double xdist = (x - origin_x);
    double ydist = (y - origin_y);
    double zdist = (z - origin_z);
    bool in = ( xdist * xdist + ydist * ydist + zdist * zdist <= radius * radius );
    return in;
}


void Inner_region_sphere::write_to_file( hid_t regions_group_id )
{
    hid_t current_region_group_id;
    herr_t status;
    int single_element = 1;
    std::string current_region_groupname = name;
    
    current_region_group_id = H5Gcreate( regions_group_id, current_region_groupname.c_str(),
					 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_region_group_id );

    status = H5LTset_attribute_string( regions_group_id, current_region_groupname.c_str(),
				       "object_type", object_type.c_str() );
    
    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "potential", &potential, single_element ); hdf5_status_check( status );

    status = H5LTset_attribute_int( regions_group_id, current_region_groupname.c_str(),
				    "total_absorbed_particles", &total_absorbed_particles, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "total_absorbed_charge", &total_absorbed_charge, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "origin_x", &origin_x, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "origin_y", &origin_y, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "origin_z", &origin_z, single_element );
    hdf5_status_check( status );

    status = H5LTset_attribute_double( regions_group_id, current_region_groupname.c_str(),
				       "radius", &radius, single_element );
    hdf5_status_check( status );
    
    status = H5Gclose(current_region_group_id); hdf5_status_check( status );
    return;
    
    // todo: separate functions
    // write_hdf5_absorbed_charge( group_id, region_name );

    // todo: call Inner_region::write_to_file()
    // to write common properties.
    // then write rest of parameters
    // write_hdf5_region_parameters( group_id, region_name );
}



// Step model

Inner_region_STEP::Inner_region_STEP( Config &conf,
				      Inner_region_STEP_config_part &inner_region_STEP_conf,
				      Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_STEP_conf );
    get_values_from_config( inner_region_STEP_conf );
    object_type = "STEP";
    total_absorbed_particles = 0;
    total_absorbed_charge = 0;
    absorbed_particles_current_timestep_current_proc = 0;
    absorbed_charge_current_timestep_current_proc = 0;
    mark_inner_nodes( spat_mesh );
    select_inner_nodes_not_at_domain_edge( spat_mesh );
    mark_near_boundary_nodes( spat_mesh );
    select_near_boundary_nodes_not_at_domain_edge( spat_mesh );
}

void Inner_region_STEP::check_correctness_of_related_config_fields(
    Config &conf,
    Inner_region_STEP_config_part &inner_region_STEP_conf )
{
  // Check if file exists?   
}

void Inner_region_STEP::get_values_from_config( Inner_region_STEP_config_part &inner_region_STEP_conf )
{
    name = inner_region_STEP_conf.inner_region_name;
    potential = inner_region_STEP_conf.inner_region_potential;
    read_geometry_file( inner_region_STEP_conf.inner_region_STEP_file );

}

void Inner_region_STEP::read_geometry_file( std::string filename )
{
  STEPControl_Reader reader;
  reader.ReadFile( filename.c_str() );
  cout << "filename = " << filename << std::endl;
  // gets the number of transferable roots
  Standard_Integer NbRoots = reader.NbRootsForTransfer();
  cout << "Number of roots in STEP file:" << NbRoots << std::endl;
  // translates all transferable roots, and returns the number of
  // successful translations
  Standard_Integer NbTrans = reader.TransferRoots();
  cout << "STEP roots transferred: " << NbTrans << std::endl;
  cout << "Number of resulting shapes is: " << reader.NbShapes() << std::endl;

  geometry = reader.OneShape();
  if (geometry.IsNull() || geometry.ShapeType() != TopAbs_SOLID) {
      std::cout << "Something wrong with model of inner_region_STEP: "
  		<< name
  		<< std::endl;
      exit( EXIT_FAILURE );
  }
}

bool Inner_region_STEP::check_if_point_inside( double x, double y, double z )
{
    gp_Pnt point(x, y, z);
    BRepClass3d_SolidClassifier solidClassifier( geometry, point, Precision::Confusion() );
    TopAbs_State in_or_out = solidClassifier.State();

    if ( in_or_out == TopAbs_OUT ){
      return false;
    }
    else if ( in_or_out == TopAbs_IN || in_or_out == TopAbs_ON ){
      return true;
    }
    else {	
      std::cout << "Unknown in_or_out state: " << in_or_out << std::endl;
      std::cout << "x=" << x << " y=" << y << " z="<< z << std::endl;
      std::cout << "Aborting.";
      std::cout << std::endl;
      std::exit( 1 );
    }
}

Inner_region_STEP::~Inner_region_STEP()
{
    // PetscErrorCode ierr;
    // ierr = VecDestroy( &phi_inside_region ); CHKERRXX( ierr );
    // ierr = VecDestroy( &rhs_inside_region ); CHKERRXX( ierr );
    // todo
}



