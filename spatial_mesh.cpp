#include "spatial_mesh.h"

Spatial_mesh::Spatial_mesh( Config &conf )
{
    check_correctness_of_related_config_fields( conf );
    init_x_grid( conf );
    init_y_grid( conf );
    init_z_grid( conf );
    allocate_ongrid_values();
    fill_node_coordinates();
    set_boundary_conditions( conf );
}


void Spatial_mesh::check_correctness_of_related_config_fields( Config &conf )
{
    grid_x_size_gt_zero( conf );
    grid_x_step_gt_zero_le_grid_x_size( conf );
    grid_y_size_gt_zero( conf );
    grid_y_step_gt_zero_le_grid_y_size( conf );
    grid_z_size_gt_zero( conf );
    grid_z_step_gt_zero_le_grid_z_size( conf );
}

void Spatial_mesh::init_x_grid( Config &conf )
{
    x_volume_size = conf.mesh_config_part.grid_x_size;
    x_n_nodes = 
	ceil( conf.mesh_config_part.grid_x_size / conf.mesh_config_part.grid_x_step ) + 1;
    x_cell_size = x_volume_size / ( x_n_nodes - 1 );
    if ( x_cell_size != conf.mesh_config_part.grid_x_step ) {
	std::cout.precision(3);
	std::cout << "X_step was shrinked to " << x_cell_size 
		  << " from " << conf.mesh_config_part.grid_x_step 
		  << " to fit round number of cells" << std::endl;
    }    
    return;
}

void Spatial_mesh::init_y_grid( Config &conf )
{
    y_volume_size = conf.mesh_config_part.grid_y_size;
    y_n_nodes = 
	ceil( conf.mesh_config_part.grid_y_size / conf.mesh_config_part.grid_y_step) + 1;
    y_cell_size = y_volume_size / ( y_n_nodes -1 );
    if ( y_cell_size != conf.mesh_config_part.grid_y_step ) {
	std::cout.precision(3);
	std::cout << "Y_step was shrinked to " << y_cell_size 
		  << " from " << conf.mesh_config_part.grid_y_step 
		  << " to fit round number of cells." << std::endl;
    }    
    return;
}

void Spatial_mesh::init_z_grid( Config &conf )
{
    z_volume_size = conf.mesh_config_part.grid_z_size;
    z_n_nodes = 
	ceil( conf.mesh_config_part.grid_z_size / conf.mesh_config_part.grid_z_step) + 1;
    z_cell_size = z_volume_size / ( z_n_nodes -1 );
    if ( z_cell_size != conf.mesh_config_part.grid_z_step ) {
	std::cout.precision(3);
	std::cout << "Z_step was shrinked to " << z_cell_size 
		  << " from " << conf.mesh_config_part.grid_z_step 
		  << " to fit round number of cells." << std::endl;
    }    
    return;
}

void Spatial_mesh::allocate_ongrid_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;
    node_coordinates.resize( boost::extents[nx][ny][nz] );
    charge_density.resize( boost::extents[nx][ny][nz] );
    potential.resize( boost::extents[nx][ny][nz] );
    electric_field.resize( boost::extents[nx][ny][nz] );

    return;
}

void Spatial_mesh::fill_node_coordinates()
{
    for ( int i = 0; i < x_n_nodes; i++ ) {
	for ( int j = 0; j < y_n_nodes; j++ ) {
	    for ( int k = 0; k < z_n_nodes; k++ ) {
		node_coordinates[i][j][k] = vec3d_init( i * x_cell_size, j * y_cell_size, k * z_cell_size );
	    }
	}
    }
}

void Spatial_mesh::clear_old_density_values()
{
    std::fill( charge_density.data(),
	       charge_density.data() + charge_density.num_elements(),
	       0.0 );

    return;
}


void Spatial_mesh::set_boundary_conditions( Config &conf )
{
    set_boundary_conditions( conf.boundary_config_part.boundary_phi_left, 
			     conf.boundary_config_part.boundary_phi_right,
			     conf.boundary_config_part.boundary_phi_top, 
			     conf.boundary_config_part.boundary_phi_bottom,
			     conf.boundary_config_part.boundary_phi_near, 
			     conf.boundary_config_part.boundary_phi_far );
}


void Spatial_mesh::set_boundary_conditions( const double phi_left, const double phi_right,
					    const double phi_top, const double phi_bottom,
					    const double phi_near, const double phi_far )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;    	

    for ( int i = 0; i < nx; i++ ) {
	for ( int k = 0; k < nz; k++ ) {
	    potential[i][0][k] = phi_bottom;
	    potential[i][ny-1][k] = phi_top;
	}
    }
    
    for ( int j = 0; j < ny; j++ ) {
	for ( int k = 0; k < nz; k++ ) {
	    potential[0][j][k] = phi_left;
	    potential[nx-1][j][k] = phi_right;
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    potential[i][j][0] = phi_near;
	    potential[i][j][nz-1] = phi_far;
	}
    }

    return;
}

void Spatial_mesh::print()
{
    print_grid();
    print_ongrid_values();
    return;
}

void Spatial_mesh::print_grid()
{
    std::cout << "Grid:" << std::endl;
    std::cout << "Length: x = " << x_volume_size << ", "
	      << "y = " << y_volume_size << ", "
	      << "z = " << z_volume_size << std::endl;
    std::cout << "Cell size: x = " << x_cell_size << ", "
	      << "y = " << y_cell_size << ", "
    	      << "z = " << z_cell_size << std::endl;
    std::cout << "Total nodes: x = " << x_n_nodes << ", "
	      << "y = " << y_n_nodes << ", "
    	      << "z = " << z_n_nodes << std::endl;
    return;
}

void Spatial_mesh::print_ongrid_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;
    std::cout << "x_node, y_node, z_node, charge_density, potential, electric_field(x,y,z)" << std::endl;
    std::cout.precision( 3 );
    std::cout.setf( std::ios::scientific );
    std::cout.fill(' ');
    std::cout.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int k = 0; k < nz; k++ ) {
		std::cout << std::setw(8) << i 
			  << std::setw(8) << j
			  << std::setw(8) << k
			  << std::setw(14) << charge_density[i][j][k]
			  << std::setw(14) << potential[i][j][k]
			  << std::setw(14) << vec3d_x( electric_field[i][j][k] ) 
			  << std::setw(14) << vec3d_y( electric_field[i][j][k] )
			  << std::setw(14) << vec3d_z( electric_field[i][j][k] ) 
			  << std::endl;
	    }
	}
    }
    return;
}

void Spatial_mesh::write_to_file_iostream( std::ofstream &output_file )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;

    output_file << "### Grid" << std::endl;
    output_file << "X volume size = " << x_volume_size << std::endl;
    output_file << "Y volume size = " << y_volume_size << std::endl;
    output_file << "Z volume size = " << z_volume_size << std::endl;
    output_file << "X cell size = " << x_cell_size << std::endl;
    output_file << "Y cell size = " << y_cell_size << std::endl;
    output_file << "Z cell size = " << z_cell_size << std::endl;
    output_file << "X nodes = " << x_n_nodes << std::endl;
    output_file << "Y nodes = " << y_n_nodes << std::endl;
    output_file << "Z nodes = " << z_n_nodes << std::endl;
    output_file << "x_node, y_node, z_node, charge_density, potential, electric_field(x,y,z)" << std::endl;
    output_file.fill(' ');
    output_file.setf( std::ios::scientific );
    output_file.precision( 2 );
    output_file.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int k = 0; k < nz; k++ ) {
		output_file << std::setw(8) << std::left << i 
			    << std::setw(8) << std::left << j
			    << std::setw(8) << std::left << k 
			    << std::setw(14) << charge_density[i][j][k]
			    << std::setw(14) << potential[i][j][k]
			    << std::setw(14) << vec3d_x( electric_field[i][j][k] ) 
			    << std::setw(14) << vec3d_y( electric_field[i][j][k] )
			    << std::setw(14) << vec3d_z( electric_field[i][j][k] ) 
			    << std::endl;
	    }
	}
    }
    return;
}

void Spatial_mesh::write_to_file_hdf5( hid_t hdf5_file_id )
{
    hid_t group_id;
    herr_t status;
    std::string hdf5_groupname = "/Spatial_mesh";
    group_id = H5Gcreate2( hdf5_file_id, hdf5_groupname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    write_hdf5_attributes( group_id );
    write_hdf5_ongrid_values( group_id );
        
    status = H5Gclose(group_id);
    return;
}

void Spatial_mesh::write_hdf5_attributes( hid_t group_id )
{
    int single_element = 1;
    std::string hdf5_current_group = "./";

    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "x_volume_size", &x_volume_size, single_element );
    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "y_volume_size", &y_volume_size, single_element );
    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "z_volume_size", &z_volume_size, single_element );
    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "x_cell_size", &x_cell_size, single_element );
    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "y_cell_size", &y_cell_size, single_element );
    H5LTset_attribute_double( group_id, hdf5_current_group.c_str(), "z_cell_size", &z_cell_size, single_element );
    H5LTset_attribute_int( group_id, hdf5_current_group.c_str(), "x_n_nodes", &x_n_nodes, single_element );
    H5LTset_attribute_int( group_id, hdf5_current_group.c_str(), "y_n_nodes", &y_n_nodes, single_element );
    H5LTset_attribute_int( group_id, hdf5_current_group.c_str(), "z_n_nodes", &z_n_nodes, single_element );
}

void Spatial_mesh::write_hdf5_ongrid_values( hid_t group_id )
{
    hid_t compound_type_for_mem, compound_type_for_file, dspace, dset;
    herr_t status;
    int rank = 1;
    hsize_t dims[rank];
    dims[0] = node_coordinates.num_elements();
    
    compound_type_for_mem = vec3d_hdf5_compound_type_for_memory();
    compound_type_for_file = vec3d_hdf5_compound_type_for_file();
    dspace = H5Screate_simple( rank, dims, NULL );

    dset = H5Dcreate( group_id, "./node_coordinates",
		      compound_type_for_file, dspace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    status = H5Dwrite( dset, compound_type_for_mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, node_coordinates.data() );
    status = H5Dclose( dset );

    H5LTmake_dataset_double( group_id, "./charge_density",
			     rank, dims, charge_density.data() );
    H5LTmake_dataset_double( group_id, "./potential",
			     rank, dims, potential.data() );

    dset = H5Dcreate( group_id, "./electric_field",
		      compound_type_for_file, dspace,
		      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    status = H5Dwrite( dset, compound_type_for_mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, electric_field.data() );
    status = H5Dclose( dset );

    status = H5Sclose( dspace );
    status = H5Tclose( compound_type_for_file );
    status = H5Tclose( compound_type_for_mem );	
}


void Spatial_mesh::grid_x_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_x_size > 0,
			   "grid_x_size < 0" );    
}

void Spatial_mesh::grid_x_step_gt_zero_le_grid_x_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_x_step > 0 ) && 
	( conf.mesh_config_part.grid_x_step <= conf.mesh_config_part.grid_x_size ),
			   "grid_x_step < 0 or grid_x_step >= grid_x_size" );    
}

void Spatial_mesh::grid_y_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_y_size > 0,
			   "grid_y_size < 0" );    
}

void Spatial_mesh::grid_y_step_gt_zero_le_grid_y_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_y_step > 0 ) && 
	( conf.mesh_config_part.grid_y_step <= conf.mesh_config_part.grid_y_size ),
			   "grid_y_step < 0 or grid_y_step >= grid_y_size" );    
}

void Spatial_mesh::grid_z_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_z_size > 0,
			   "grid_z_size < 0" );    
}

void Spatial_mesh::grid_z_step_gt_zero_le_grid_z_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_z_step > 0 ) && 
	( conf.mesh_config_part.grid_z_step <= conf.mesh_config_part.grid_z_size ),
			   "grid_z_step < 0 or grid_z_step >= grid_z_size" );    
}


void Spatial_mesh::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " << message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}

double Spatial_mesh::node_number_to_coordinate_x( int i )
{
    if( i >= 0 && i < x_n_nodes ){
	return i*x_cell_size; 
    } else {
	printf( "invalid node number i=%d at node_number_to_coordinate_x\n", i );
	exit( EXIT_FAILURE );
    }
}

double Spatial_mesh::node_number_to_coordinate_y( int j )
{
    if( j >= 0 && j < y_n_nodes ){
	return j*y_cell_size; 
    } else {
	printf( "invalid node number j=%d at node_number_to_coordinate_y\n", j );
	exit( EXIT_FAILURE );
    }
}

double Spatial_mesh::node_number_to_coordinate_z( int k )
{
    if( k >= 0 && k < z_n_nodes ){
	return k*z_cell_size; 
    } else {
	printf( "invalid node number k=%d at node_number_to_coordinate_z\n", k );
	exit( EXIT_FAILURE );
    }
}


Spatial_mesh::~Spatial_mesh() {}
