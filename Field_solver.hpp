#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <gsl/gsl_linalg.h>
#include "Spatial_mesh.hpp"

template< int dim >
class Field_solver {
  public:
    Field_solver( Spatial_mesh<dim> &spat_mesh );
    void eval_potential( Spatial_mesh<dim> &spat_mesh );
    void eval_fields_from_potential( Spatial_mesh<dim> &spat_mesh );
    virtual ~Field_solver();
  private:
    gsl_matrix *a;
    gsl_vector *rhs;
    gsl_vector *phi_vec;
    gsl_permutation *pmt;
    int perm_sign;
    gsl_matrix* construct_equation_matrix( int nx, int ny, double dx, double dy );
    gsl_matrix* construct_d2dx2_in_2d( int nx, int ny );
    gsl_matrix* construct_d2dy2_in_2d( int nx, int ny );
    // Solve potential
    void solve_poisson_eqn( Spatial_mesh<dim> &spat_mesh );
    void init_rhs_vector( Spatial_mesh<dim> &spat_mesh );
    int kronecker_delta( int i,  int j );
    void transfer_solution_to_spat_mesh( Spatial_mesh<dim> &spat_mesh );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

template< int dim >
Field_solver<dim>::Field_solver( Spatial_mesh<dim> &spat_mesh )
{
    std::cout << "Unsupported dim=" << dim << " in Field_solver. Aborting.";
    exit( EXIT_FAILURE );
}


template<>
Field_solver<2>::Field_solver( Spatial_mesh<2> &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int nrow = (nx-2)*(ny-2);
    
    a = construct_equation_matrix( nx, ny, dx, dy );    
    rhs = gsl_vector_alloc( nrow );
    phi_vec = gsl_vector_alloc( nrow );
    
    pmt = gsl_permutation_alloc( nrow );    
    gsl_linalg_LU_decomp( a, pmt, &perm_sign );    
}


template<>
gsl_matrix* Field_solver<2>::construct_equation_matrix( int nx, int ny, double dx, double dy )
{
    gsl_matrix *a = construct_d2dx2_in_2d( nx, ny );
    gsl_matrix_scale( a, dy * dy );

    gsl_matrix *d2dy2 = construct_d2dy2_in_2d( nx, ny );
    gsl_matrix_scale( d2dy2, dx * dx ); 
    
    gsl_matrix_add( a, d2dy2 );
    gsl_matrix_free( d2dy2 );

    return a;
}


template< int dim >
gsl_matrix* Field_solver<dim>::construct_d2dx2_in_2d( int nx, int ny )
{
  int nrow = ( nx - 2 ) * ( ny - 2 );
  int ncol = nrow;
  gsl_matrix *d2dx2 = gsl_matrix_alloc( nrow, ncol );
  gsl_matrix_set_zero( d2dx2 );
  
  // first construct tridiagonal matrix, then set some
  // boundary elements to zero

  for( int i = 0; i < nrow; i++ ) {
    for( int j = 0; j < ncol; j++ ){
      if ( i == j ){
	gsl_matrix_set( d2dx2, i, j, -2.0 );
      } else if ( j + 1 == i || j - 1 == i ) {
	gsl_matrix_set( d2dx2, i, j, 1.0 );
      }
    }
  }
  
  for( int i = 0; i < nrow; i++ ) {
      for( int j = 0; j < ncol; j++ ){
  	  if ( ( j - 1 == i ) && ( j % ( nx - 2 ) == 0 ) ){
  	      gsl_matrix_set( d2dx2, i, j, 0 );
  	  } else if ( j + 1 == i && ( ( j + 1 ) % ( nx - 2 ) == 0 ) ) {
  	      gsl_matrix_set( d2dx2, i, j, 0 );
  	  }
      }
  }
  
  return d2dx2;
}


template< int dim >
gsl_matrix* Field_solver<dim>::construct_d2dy2_in_2d( int nx, int ny )
{
  int nrow = ( nx - 2 ) * ( ny - 2 );
  int ncol = nrow;
  gsl_matrix *d2dy2 = gsl_matrix_alloc( nrow, ncol );
  gsl_matrix_set_zero( d2dy2 );
  
  for( int i = 0; i < nrow; i++ ) {
    for( int j = 0; j < ncol; j++ ){
      if ( i == j ){
	gsl_matrix_set( d2dy2, i, j, -2.0 );
      } else if ( j + (nx - 2) == i || j - (nx - 2) == i ) {
	gsl_matrix_set( d2dy2, i, j, 1.0 );
      }
    }
  }
          
  return d2dy2;
}

template< int dim >
void Field_solver<dim>::eval_potential( Spatial_mesh<dim> &spat_mesh )
{
    solve_poisson_eqn( spat_mesh );
}

template< int dim >
void Field_solver<dim>::solve_poisson_eqn( Spatial_mesh<dim> &spat_mesh )
{
    int ierror;
    init_rhs_vector( spat_mesh );

    ierror = gsl_linalg_LU_solve( a, pmt, rhs, phi_vec );	    
    if ( ierror ){
	printf( "Error while solving Poisson equation (gsl_linalg_LU_solve). \n" );
	printf( "ierror = %d \n", ierror );
    	exit( EXIT_FAILURE );
    }

    transfer_solution_to_spat_mesh( spat_mesh );
    
    return;
}


template< int dim >
void Field_solver<dim>::init_rhs_vector( Spatial_mesh<dim> &spat_mesh )
{
    std::cout << "Unsupported dim=" << dim << " "
	      << "in Field_solver::init_rhs_vector. Aborting." << std::endl;
    exit( EXIT_FAILURE );    
}


template<>
void Field_solver<2>::init_rhs_vector( Spatial_mesh<2> &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    //int nrow = (nx-2)*(ny-2);
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;    
    double rhs_at_node;

    // todo: split into separate functions
    // start processing rho from the top left corner
    for ( int j = ny-2; j >= 1; j-- ) { 
	for ( int i = 1; i <= nx-2; i++ ) {
	    // - 4 * pi * rho * dx^2 * dy^2
	    rhs_at_node = -4.0 * M_PI * spat_mesh.charge_density[i][j];
	    rhs_at_node = rhs_at_node * dx * dx * dy * dy;
	    // left and right boundary
	    rhs_at_node = rhs_at_node
		- dy * dy *
		( kronecker_delta(i,1) * spat_mesh.potential[0][j] +
		  kronecker_delta(i,nx-2) * spat_mesh.potential[nx-1][j] );
	    // top and bottom boundary
	    rhs_at_node = rhs_at_node
		- dx * dx *
		( kronecker_delta(j,1) * spat_mesh.potential[i][0] +
		  kronecker_delta(j,ny-2) * spat_mesh.potential[i][ny-1] );
	    // set rhs vector values
	    // todo: separate function for: (i - 1) + ( ( ny - 2 ) - j ) * (nx-2)
	    gsl_vector_set( rhs, (i - 1) + ( ( ny - 2 ) - j ) * (nx-2), rhs_at_node );
	}
    }    
}


template< int dim >
int Field_solver<dim>::kronecker_delta( int i,  int j )
{
    if ( i == j ) {
	return 1;
    } else {
	return 0;
    }
}


template< int dim >
void Field_solver<dim>::transfer_solution_to_spat_mesh( Spatial_mesh<dim> &spat_mesh )
{
    std::cout << "Unsupported dim=" << dim << " "
	      << "in Field_solver::init_rhs_vector. Aborting." << std::endl;
    exit( EXIT_FAILURE );
}


template<>
void Field_solver<2>::transfer_solution_to_spat_mesh( Spatial_mesh<2> &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    //int nrow = (nx-2)*(ny-2);
    
    for ( int j = ny-2; j >= 1; j-- ) { 
	for ( int i = 1; i <= nx-2; i++ ) {
	    spat_mesh.potential[i][j] =
		gsl_vector_get( phi_vec, (i - 1) + ( ( ny - 2 ) - j ) * (nx-2) );
	}
    }    
}


template< int dim >
void Field_solver<dim>::eval_fields_from_potential( Spatial_mesh<dim> &spat_mesh )
{
    std::cout << "Unsupported dim=" << dim << " "
	      << "in Field_solver::init_rhs_vector. Aborting." << std::endl;
    exit( EXIT_FAILURE );
}


template<>
void Field_solver<2>::eval_fields_from_potential( Spatial_mesh<2> &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double **phi = spat_mesh.potential;
    double ex[nx][ny], ey[nx][ny];

    for ( int j = 0; j < ny; j++ ) {
	for ( int i = 0; i < nx; i++ ) {
	    if ( i == 0 ) {
		ex[i][j] = - boundary_difference( phi[i][j], phi[i+1][j], dx );
	    } else if ( i == nx-1 ) {
		ex[i][j] = - boundary_difference( phi[i-1][j], phi[i][j], dx );
	    } else {
		ex[i][j] = - central_difference( phi[i-1][j], phi[i+1][j], dx );
	    }
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    if ( j == 0 ) {
		ey[i][j] = - boundary_difference( phi[i][j], phi[i][j+1], dy );
	    } else if ( j == ny-1 ) {
		ey[i][j] = - boundary_difference( phi[i][j-1], phi[i][j], dy );
	    } else {
		ey[i][j] = - central_difference( phi[i][j-1], phi[i][j+1], dy );
	    }
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    spat_mesh.electric_field[i][j] = VecNd<2>( ex[i][j], ey[i][j] );
	}
    }

    return;
}


template< int dim >
double Field_solver<dim>::central_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}


template< int dim >
double Field_solver<dim>::boundary_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / dx );
}


template< int dim >
Field_solver<dim>::~Field_solver()
{    
    gsl_permutation_free( pmt );
    gsl_matrix_free( a );
    gsl_vector_free( rhs );
    gsl_vector_free( phi_vec );
}

#endif /* _FIELD_SOLVER_H_ */
