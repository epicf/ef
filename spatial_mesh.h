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
    double x_volume_size, y_volume_size;
    double x_cell_size, y_cell_size;
    int x_n_nodes, y_n_nodes;
    ArrayNd< dim, double > charge_density;
    ArrayNd< dim, double > potential;
    ArrayNd< dim, VecNd< dim > > electric_field;
  public:
    Spatial_mesh( Config &conf );
    void clear_old_density_values();
    void set_boundary_conditions( Config &conf );
    void print();
    void write_to_file( std::ofstream &output_file );
    virtual ~Spatial_mesh();
  private:
    // init
    void check_correctness_of_related_config_fields( Config &conf );
    void init_x_grid( Config &conf );
    void init_y_grid( Config &conf );
    void allocate_ongrid_values( );
    void set_boundary_conditions( const double phi_left, const double phi_right,
				  const double phi_top, const double phi_bottom )
    // print
    void print_grid();
    void print_ongrid_values();
    // config check
    void grid_x_size_gt_zero( Config &conf );
    void grid_x_step_gt_zero_le_grid_x_size( Config &conf );
    void grid_y_size_gt_zero( Config &conf );
    void grid_y_step_gt_zero_le_grid_y_size( Config &conf );
    void check_and_exit_if_not( const bool &should_be, const std::string &message );
};

template< int dim >
Spatial_mesh<dim>::Spatial_mesh( Config &conf )
{
    check_correctness_of_related_config_fields( conf );

    // todo: make it a separate function?
    if( dim >= 1 && dim <= 3 ){
	init_x_grid( conf );
    } else if( dim >= 2 && dim <= 3 ) {
	init_y_grid( conf );
    } else if( dim == 3 ) {
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
    if( dim >= 1 && dim <= 3 ){
	grid_x_size_gt_zero( conf );
	grid_x_step_gt_zero_le_grid_x_size( conf );
    } else if( dim >= 2 && dim <= 3 ) {
	grid_y_size_gt_zero( conf );
	grid_y_step_gt_zero_le_grid_y_size( conf );
    } else if( dim == 3 ) {
	grid_z_size_gt_zero( conf );
	grid_z_step_gt_zero_le_grid_z_size( conf );
    } else {
	std::cout << "Unsupported dim=" << dim << " in Spatial_mesh. Aborting.";
	exit( EXIT_FAILURE );
    }
}

// todo: merge init_{x,y,z}_grid into single function
// too much repetition
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

// template< int dim >
// void Spatial_mesh<dim>::allocate_ongrid_values( )
// {
//     int nx = x_n_nodes;
//     int ny = y_n_nodes;    
//     charge_density = (double **) malloc( nx * sizeof(double *) );
//     potential = (double **) malloc( nx * sizeof(double *) );
//     electric_field = (VecNd<2> **) malloc( nx * sizeof(VecNd<2> *) );
//     if ( ( charge_density == NULL ) || 
// 	 ( potential == NULL ) || 
// 	 ( electric_field == NULL ) ) {
// 	std::cout << "allocate_arrays_for_ongrid_values: rows: out of memory " 
// 		  << std::endl;
// 	exit( EXIT_FAILURE );	
//     }
//     for( int i = 0; i < nx; i++) {
// 	charge_density[i] = (double *) calloc( ny, sizeof(double) );
// 	potential[i] = (double *) calloc( ny, sizeof(double) );
// 	electric_field[i] = (VecNd<2> *) calloc( ny, sizeof(VecNd<2>) );
// 	if ( ( charge_density[i] == NULL ) || 
// 	     ( potential[i] == NULL ) || 
// 	     ( electric_field[i] == NULL ) ) {
// 	    std::cout << "allocate_arrays_for_ongrid_values: cols: out of memory " 
// 		      << std::endl;
// 	    exit( EXIT_FAILURE );	
// 	}
//     }
//     return;
// }

template< int dim >
void Spatial_mesh<dim>::clear_old_density_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;    

    // надо сделать итератор по всему ArrayNd.
    // тогда будет нормально.
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    charge_density[i][j] = 0;
	}
    }
}

// template< int dim >
// void Spatial_mesh::set_boundary_conditions( Config &conf )
// {
//     set_boundary_conditions( conf.boundary_config_part.boundary_phi_left, 
// 			     conf.boundary_config_part.boundary_phi_right,
// 			     conf.boundary_config_part.boundary_phi_top, 
// 			     conf.boundary_config_part.boundary_phi_bottom );
// }


// void Spatial_mesh::set_boundary_conditions( const double phi_left, const double phi_right,
// 					    const double phi_top, const double phi_bottom )
// {
//     int nx = x_n_nodes;
//     int ny = y_n_nodes;    

//     for ( int i = 0; i < nx; i++ ) {
// 	potential[i][0] = phi_bottom;
// 	potential[i][ny-1] = phi_top;
//     }
    
//     for ( int j = 0; j < ny; j++ ) {
// 	potential[0][j] = phi_left;
// 	potential[nx-1][j] = phi_right;
//     }

//     return;
// }

template< int dim >
void Spatial_mesh<dim>::print()
{
    print_grid();
    print_ongrid_values();
    return;
}

// template< int dim >
// void Spatial_mesh<dim>::print_grid()
// {
//     std::cout << "Grid:" << std::endl;
//     std::cout << "Length: x = " << x_volume_size << ", "
// 	      << "y = " << y_volume_size << std::endl;
//     std::cout << "Cell size: x = " << x_cell_size << ", "
// 	      << "y = " << y_cell_size << std::endl;
//     std::cout << "Total nodes: x = " << x_n_nodes << ", "
// 	      << "y = " << y_n_nodes << std::endl;
//     return;
// }

template< int dim >
void Spatial_mesh<dim>::print_ongrid_values()
{
    int nx = x_n_nodes;
    int ny = y_n_nodes;
    std::cout << "x_node \t\t y_node  charge_density   potential \t electric_field(x,y)" << std::endl;
    std::cout.precision( 3 );
    std::cout.setf( std::ios::scientific );
    std::cout.fill(' ');
    std::cout.setf( std::ios::right );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    std::cout << std::setw(8) << i 
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


#endif /* _SPATIAL_MESH_H_ */
