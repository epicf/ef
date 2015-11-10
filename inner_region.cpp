#include "inner_region.h"

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



// Box

Inner_region_box::Inner_region_box( Config &conf,
				    Inner_region_box_config_part &inner_region_box_conf,
				    Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_box_conf );
    get_values_from_config( inner_region_box_conf );
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



// Sphere

Inner_region_sphere::Inner_region_sphere( Config &conf,
					  Inner_region_sphere_config_part &inner_region_sphere_conf,
					  Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_sphere_conf );
    get_values_from_config( inner_region_sphere_conf );
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


// Step model

Inner_region_STEP::Inner_region_STEP( Config &conf,
				      Inner_region_STEP_config_part &inner_region_STEP_conf,
				      Spatial_mesh &spat_mesh )
{
    check_correctness_of_related_config_fields( conf, inner_region_STEP_conf );
    get_values_from_config( inner_region_STEP_conf );
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
    // todo
}
