#ifndef _SPATIAL_MESH_H_
#define _SPATIAL_MESH_H_

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include "config.h"
#include "VecNd.hpp"

template< int dim >
class Spatial_mesh {
  public:
    double x_volume_size, y_volume_size, z_volume_size;
    double x_cell_size, y_cell_size, z_cell_size;
    int x_n_nodes, y_n_nodes, z_n_nodes;
    ArrayNd< dim, double > charge_density;
    ArrayNd< dim, double > potential;
    ArrayNd< dim, VecNd< dim > > electric_field;
  public:
    Spatial_mesh( Config &conf );
    void clear_old_density_values();
    void set_boundary_conditions( Config &conf );
    void print( std::ofstream &out_stream );
    virtual ~Spatial_mesh();
  private:
    // init
    void check_correctness_of_related_config_fields( Config &conf );
    void init_x_grid( Config &conf );
    void init_y_grid( Config &conf );
    void init_z_grid( Config &conf );
    void allocate_ongrid_values();
    void set_boundary_conditions( Config &conf );
    void set_boundary_conditions_1d( const double phi_left, const double phi_right );
    void set_boundary_conditions_2d( const double phi_left, const double phi_right,
				     const double phi_top, const double phi_bottom );
    void set_boundary_conditions_3d( const double phi_left, const double phi_right,
				     const double phi_top, const double phi_bottom,
				     const double phi_near, const double phi_far );
    // print
    void print( std::ofstream &out_stream );
    void print_1d( std::ofstream &out_stream );
    void print_2d( std::ofstream &out_stream );
    void print_3d( std::ofstream &out_stream );
    // config check
    void grid_x_size_gt_zero( Config &conf );
    void grid_x_step_gt_zero_le_grid_x_size( Config &conf );
    void grid_y_size_gt_zero( Config &conf );
    void grid_y_step_gt_zero_le_grid_y_size( Config &conf );
    void grid_z_size_gt_zero( Config &conf );
    void grid_z_step_gt_zero_le_grid_y_size( Config &conf );
    void check_and_exit_if_not( const bool &should_be, const std::string &message );
};

template< int dim >
Spatial_mesh<dim>::Spatial_mesh( Config &conf )
{
    check_correctness_of_related_config_fields( conf );

    // todo: make it a separate function?
    // put else first, then 3 separate ifs?
    if( dim == 1 )
	init_x_grid( conf );
    } else if( dim == 2 ) {
	init_x_grid( conf );
	init_y_grid( conf );
    } else if( dim == 3 ) {
	init_x_grid( conf );
	init_y_grid( conf );
	init_z_grid( conf );
    } else {
	std::cout << "Unsupported dim=" << dim << " in Spatial_mesh. Aborting.";
	exit( EXIT_FAILURE );
    }

    allocate_ongrid_values();
    set_boundary_conditions( conf );
}

template< int dim >
void Spatial_mesh<dim>::check_correctness_of_related_config_fields( Config &conf )
{
    // put dim < 1 && dim > 3 check first and
    // then 3 separate if's?
    if( dim == 1 ){
	grid_x_size_gt_zero( conf );
	grid_x_step_gt_zero_le_grid_x_size( conf );
    } else if( dim == 2 ) {
	grid_x_size_gt_zero( conf );
	grid_x_step_gt_zero_le_grid_x_size( conf );
	grid_y_size_gt_zero( conf );
	grid_y_step_gt_zero_le_grid_y_size( conf );
    } else if( dim == 3 ) {
	grid_x_size_gt_zero( conf );
	grid_x_step_gt_zero_le_grid_x_size( conf );
	grid_y_size_gt_zero( conf );
	grid_y_step_gt_zero_le_grid_y_size( conf );
	grid_z_size_gt_zero( conf );
	grid_z_step_gt_zero_le_grid_z_size( conf );
    } else {
	std::cout << "Unsupported dim=" << dim << " in Spatial_mesh. Aborting.";
	exit( EXIT_FAILURE );
    }
}

// todo: merge init_{x,y,z}_grid into single function
// too much repetition
// Guess it won't be possible without С macroprogramming. Which is even worse.
template< int dim >
void Spatial_mesh<dim>::init_x_grid( Config &conf )
{
    x_volume_size = conf.mesh_config_part.grid_x_size;
    x_n_nodes = 
	ceil( conf.mesh_config_part.grid_x_size / conf.mesh_config_part.grid_x_step ) + 1;
    x_cell_size = x_volume_size / ( x_n_nodes - 1 );
    if ( x_cell_size != conf.mesh_config_part.grid_x_step ) {
	std::cout.setf( std::ios::scientific );
	std::cout.precision(3);
	std::cout << "X_step was shrinked to " << x_cell_size 
		  << " from " << conf.mesh_config_part.grid_x_step 
		  << " to fit round number of cells" << std::endl;
    }    
    return;
}

template< int dim >
void Spatial_mesh<dim>::init_y_grid( Config &conf )
{
    y_volume_size = conf.mesh_config_part.grid_y_size;
    y_n_nodes = 
	ceil( conf.mesh_config_part.grid_y_size / conf.mesh_config_part.grid_y_step) + 1;
    y_cell_size = y_volume_size / ( y_n_nodes -1 );
    if ( y_cell_size != conf.mesh_config_part.grid_y_step ) {
	std::cout.setf( std::ios::scientific );
	std::cout.precision(3);
	std::cout << "Y_step was shrinked to " << y_cell_size 
		  << " from " << conf.mesh_config_part.grid_y_step 
		  << " to fit round number of cells." << std::endl;
    }    
    return;
}

template< int dim >
void Spatial_mesh<dim>::init_z_grid( Config &conf )
{
    z_volume_size = conf.mesh_config_part.grid_z_size;
    z_n_nodes = 
	ceil( conf.mesh_config_part.grid_z_size / conf.mesh_config_part.grid_z_step) + 1;
    z_cell_size = z_volume_size / ( z_n_nodes -1 );
    if ( z_cell_size != conf.mesh_config_part.grid_z_step ) {
	std::cout.setf( std::ios::scientific );
	std::cout.precision(3);
	std::cout << "Z_step was shrinked to " << z_cell_size 
		  << " from " << conf.mesh_config_part.grid_z_step 
		  << " to fit round number of cells." << std::endl;
    }    
    return;
}

template< int dim >
void Spatial_mesh<dim>::allocate_ongrid_values()
{
    if( dim == 1 ){
	// is it necessary to zero-fill the arrays?
	charge_density.allocate( x_n_nodes );
	potential.allocate( x_n_nodes );
	electric_field.allocate( x_n_nodes );
    } else if( dim == 2 ) {
	charge_density.allocate( x_n_nodes, y_n_nodes );
	potential.allocate( x_n_nodes, y_n_nodes );
	electric_field.allocate( x_n_nodes, y_n_nodes );
    } else if( dim == 3 ) {
	charge_density.allocate( x_n_nodes, y_n_nodes, z_n_nodes );
	potential.allocate( x_n_nodes, y_n_nodes, z_n_nodes );
	electric_field.allocate( x_n_nodes, y_n_nodes, z_n_nodes );
    } else {
	std::cout << "Unsupported dim=" << dim << " in Spatial_mesh. Aborting.";
	exit( EXIT_FAILURE );
    }
    return;
}


template< int dim >
void Spatial_mesh<dim>::clear_old_density_values()
{
    for( auto &rho : charge_density ){
	rho = 0;
    }
}

template< int dim >
void Spatial_mesh<dim>::set_boundary_conditions( Config &conf )
{
    if( dim == 1 ){
	set_boundary_conditions_1d( conf.boundary_config_part.boundary_phi_left, 
				    conf.boundary_config_part.boundary_phi_right );
    } else if( dim == 2 ){
	set_boundary_conditions_2d( conf.boundary_config_part.boundary_phi_left, 
				    conf.boundary_config_part.boundary_phi_right,
				    conf.boundary_config_part.boundary_phi_top, 
				    conf.boundary_config_part.boundary_phi_bottom );
    } else if( dim == 3 ){
	set_boundary_conditions_3d( conf.boundary_config_part.boundary_phi_left, 
				    conf.boundary_config_part.boundary_phi_right,
				    conf.boundary_config_part.boundary_phi_top, 
				    conf.boundary_config_part.boundary_phi_bottom,
				    conf.boundary_config_part.boundary_phi_near, 
				    conf.boundary_config_part.boundary_phi_far );
    } else {
	std::cout << "Unsupported dim = " << dim
		  << " in Spatial_mesh<dim>::print_ongrid_values. Aborting.";
	exit( EXIT_FAILURE );
    }
    return;

}

template< int dim >
void Spatial_mesh<dim>::set_boundary_conditions_1d( const double phi_left, const double phi_right,
						    const double phi_top, const double phi_bottom )
{
    int nx = x_n_nodes;

    potential[0] = phi_left;
    potential[nx-1] = phi_right;

    return;
}

template< int dim >
void Spatial_mesh<dim>::set_boundary_conditions_2d( const double phi_left, const double phi_right,
						    const double phi_top, const double phi_bottom )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;    

    for ( int i = 0; i < nx; i++ ) {
	potential[i][0] = phi_bottom;
	potential[i][ny-1] = phi_top;
    }
    
    for ( int j = 0; j < ny; j++ ) {
	potential[0][j] = phi_left;
	potential[nx-1][j] = phi_right;
    }

    return;
}

template< int dim >
void Spatial_mesh<dim>::set_boundary_conditions_3d( const double phi_left, const double phi_right,
						    const double phi_top, const double phi_bottom,
						    const double phi_near, const double phi_far )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;         

    for ( int j = 0; j < ny; j++ ) {
        for ( int k = 0; k < nz; k++ ) {
            potential[0][j][k] = phi_left;
            potential[nx-1][j][k] = phi_right;
        }
    }
    
    for ( int i = 0; i < nx; i++ ) {
        for ( int k = 0; k < nz; k++ ) {
            potential[i][0][k] = phi_bottom;
            potential[i][ny-1][k] = phi_top;
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

template< int dim >
void Spatial_mesh<dim>::print( std::ofstream &out_stream )
{
    if( dim == 1 ){
	print_1d( out_stream );
    } else if( dim == 2 ){
	print_2d( out_stream );
    } else if( dim == 3 ){
	print_3d( out_stream );
    } else {
	std::cout << "Unsupported dim = " << dim
		  << " in Spatial_mesh<dim>::print_ongrid_values. Aborting.";
	exit( EXIT_FAILURE );
    }
    return;
}

template< int dim >
void Spatial_mesh<dim>::print_1d( std::ofstream &out_stream )
{
    int nx = x_n_nodes;
    out_stream << "### Grid" << std::endl;
    out_stream << "X volume size = " << x_volume_size << std::endl;
    out_stream << "X cell size = " << x_cell_size << std::endl;
    out_stream << "X nodes = " << x_n_nodes << std::endl;
    out_stream << "x_node charge_density potential electric_field(x)" << std::endl;
    out_stream.precision( 3 );
    out_stream.setf( std::ios::scientific );
    out_stream.fill(' ');
    out_stream.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	out_stream << std::setw(8) << i 
		   << std::setw(14) << charge_density[i]
		   << std::setw(14) << potential[i]
		   << std::setw(14) << electric_field[i].x()
		   << std::endl;
    }
    return;
}

template< int dim >
void Spatial_mesh<dim>::print_2d( std::ofstream &out_stream )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    out_stream << "### Grid" << std::endl;
    out_stream << "X volume size = " << x_volume_size << std::endl;
    out_stream << "Y volume size = " << y_volume_size << std::endl;
    out_stream << "X cell size = " << x_cell_size << std::endl;
    out_stream << "Y cell size = " << y_cell_size << std::endl;
    out_stream << "X nodes = " << x_n_nodes << std::endl;
    out_stream << "Y nodes = " << y_n_nodes << std::endl;
    out_stream << "x_node y_node charge_density potential electric_field(x,y)" << std::endl;
    out_stream.precision( 3 );
    out_stream.setf( std::ios::scientific );
    out_stream.fill(' ');
    out_stream.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    out_stream << std::setw(8) << i 
		       << std::setw(8) << j 
		       << std::setw(14) << charge_density[i][j]
		       << std::setw(14) << potential[i][j]
		       << std::setw(14) << electric_field[i][j].x()
		       << std::setw(14) << electric_field[i][j].y()
		       << std::endl;
	}
    }
    return;
}

template< int dim >
void Spatial_mesh<dim>::print_3d( std::ofstream &out_stream )
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    int nz = z_n_nodes;
    out_stream << "### Grid" << std::endl;
    out_stream << "X volume size = " << x_volume_size << std::endl;
    out_stream << "Y volume size = " << y_volume_size << std::endl;
    out_stream << "Z volume size = " << z_volume_size << std::endl;
    out_stream << "X cell size = " << x_cell_size << std::endl;
    out_stream << "Y cell size = " << y_cell_size << std::endl;
    out_stream << "Z cell size = " << z_cell_size << std::endl;
    out_stream << "X nodes = " << x_n_nodes << std::endl;
    out_stream << "Y nodes = " << y_n_nodes << std::endl;
    out_stream << "Z nodes = " << z_n_nodes << std::endl;
    out_stream << "x_node y_node z_node" << " "
	       << "charge_density potential electric_field(x,y,z)" << std::endl;
    out_stream.precision( 3 );
    out_stream.setf( std::ios::scientific );
    out_stream.fill(' ');
    out_stream.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int k = 0; k < nz; k++ ) {
		out_stream << std::setw(8) << i 
			   << std::setw(8) << j
			   << std::setw(8) << k 
			   << std::setw(14) << charge_density[i][j][k]
			   << std::setw(14) << potential[i][j][k]
			   << std::setw(14) << electric_field[i][j][k].x()
			   << std::setw(14) << electric_field[i][j][k].y()
			   << std::setw(14) << electric_field[i][j][k].z()
			   << std::endl;
	    }
	}
    }
    return;
}

template< int dim >
void Spatial_mesh<dim>::grid_x_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_x_size > 0,
			   "grid_x_size < 0" );
}

template< int dim >
void Spatial_mesh<dim>::grid_x_step_gt_zero_le_grid_x_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_x_step > 0 ) && 
	( conf.mesh_config_part.grid_x_step <= conf.mesh_config_part.grid_x_size ),
	"grid_x_step < 0 or grid_x_step >= grid_x_size" );
}

template< int dim >
void Spatial_mesh<dim>::grid_y_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_y_size > 0,
			   "grid_y_size < 0" );
}

template< int dim >
void Spatial_mesh<dim>::grid_y_step_gt_zero_le_grid_y_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_y_step > 0 ) && 
	( conf.mesh_config_part.grid_y_step <= conf.mesh_config_part.grid_y_size ),
	"grid_y_step < 0 or grid_y_step >= grid_y_size" );
}

template< int dim >
void Spatial_mesh<dim>::grid_z_size_gt_zero( Config &conf )
{
    check_and_exit_if_not( conf.mesh_config_part.grid_z_size > 0,
			   "grid_z_size < 0" );
}

template< int dim >
void Spatial_mesh<dim>::grid_z_step_gt_zero_le_grid_z_size( Config &conf )
{
    check_and_exit_if_not( 
	( conf.mesh_config_part.grid_z_step > 0 ) && 
	( conf.mesh_config_part.grid_z_step <= conf.mesh_config_part.grid_z_size ),
	"grid_z_step < 0 or grid_z_step >= grid_z_size" );
}

template< int dim >
void Spatial_mesh<dim>::check_and_exit_if_not( const bool &should_be, const std::string &message )
{
    if( !should_be ){
	std::cout << "Error: " << message << std::endl;
	exit( EXIT_FAILURE );
    }
    return;
}

template< int dim >
Spatial_mesh<dim>::~Spatial_mesh()
{
    // Free ArrayNd?
    // Dont necessary. Should be done in ArrayNd destructor.
}


#endif /* _SPATIAL_MESH_H_ */
