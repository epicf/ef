#include "field_solver.h"

Field_solver::Field_solver( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int nrow = (nx-2)*(ny-2)*(nz-2);
    
    a = construct_equation_matrix( nx, ny, nz, dx, dy, dz );    
    rhs = gsl_vector_alloc( nrow );
    phi_vec = gsl_vector_alloc( nrow );
    
    pmt = gsl_permutation_alloc( nrow );    
    gsl_linalg_LU_decomp( a, pmt, &perm_sign );    
}

gsl_matrix* Field_solver::construct_equation_matrix( int nx, int ny, int nz,
						     double dx, double dy, double dz )
{
    gsl_matrix *a = construct_d2dx2_in_3d( nx, ny, nz );
    gsl_matrix_scale( a, dy * dy * dz * dz );

    gsl_matrix *d2dy2 = construct_d2dy2_in_3d( nx, ny, nz );
    gsl_matrix_scale( d2dy2, dx * dx * dz * dz ); 
    
    gsl_matrix_add( a, d2dy2 );
    gsl_matrix_free( d2dy2 );
    
    gsl_matrix *d2dz2 = construct_d2dz2_in_3d( nx, ny, nz );
    gsl_matrix_scale( d2dz2, dx * dx * dy * dy );     

    gsl_matrix_add( a, d2dz2 );
    gsl_matrix_free( d2dz2 );
    
    return a;
}

gsl_matrix* Field_solver::construct_d2dx2_in_3d( int nx, int ny, int nz )
{
    gsl_matrix *d2dx2_2d = construct_d2dx2_in_2d( nx, ny );
    gsl_matrix *d2dx2_3d = multiply_pattern_along_diagonal( d2dx2_2d, (nx-2)*(ny-2), nz-2 );
    gsl_matrix_free( d2dx2_2d );
    
    return d2dx2_3d;
}

gsl_matrix* Field_solver::construct_d2dy2_in_3d( int nx, int ny, int nz )
{
    gsl_matrix *d2dy2_2d = construct_d2dy2_in_2d( nx, ny );
    gsl_matrix *d2dy2_3d = multiply_pattern_along_diagonal( d2dy2_2d, (nx-2)*(ny-2), nz-2 );
    gsl_matrix_free( d2dy2_2d );
    
    return d2dy2_3d;
}

gsl_matrix* Field_solver::construct_d2dz2_in_3d( int nx, int ny, int nz )
{
    int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    int ncol = nrow;
    gsl_matrix *d2dz2_3d = gsl_matrix_alloc( nrow, ncol );
    gsl_matrix_set_zero( d2dz2_3d );
  
    for( int i = 0; i < nrow; i++ ) {
	for( int j = 0; j < ncol; j++ ){
	    if ( i == j ){
		gsl_matrix_set( d2dz2_3d, i, j, -2.0 );
	    } else if ( j + (nx - 2)*(ny - 2) == i || j - (nx - 2)*(ny - 2) == i ) {
		gsl_matrix_set( d2dz2_3d, i, j, 1.0 );
	    }
	}
    }
              
    return d2dz2_3d;
}

gsl_matrix* Field_solver::multiply_pattern_along_diagonal( gsl_matrix *pattern, int pt_size, int n_times )
{
    int mul_nrow = pt_size * n_times;
    int mul_ncol = mul_nrow;
    gsl_matrix *mul = gsl_matrix_alloc( mul_nrow, mul_ncol );
    gsl_matrix_set_zero( mul );

    for( int i = 0; i < mul_nrow; i++ ) {
	for( int j = 0; j < mul_ncol; j++ ){
	    if( i / pt_size == j / pt_size ) {
		// on diagonal inside pattern matrix		
		gsl_matrix_set( mul, i, j, gsl_matrix_get( pattern, i%pt_size, j%pt_size )  );
	    }
	}
    }
    
    return mul;
}

gsl_matrix* Field_solver::construct_d2dx2_in_2d( int nx, int ny )
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


gsl_matrix* Field_solver::construct_d2dy2_in_2d( int nx, int ny )
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

void Field_solver::eval_potential( Spatial_mesh &spat_mesh )
{
    solve_poisson_eqn( spat_mesh );
}

void Field_solver::solve_poisson_eqn( Spatial_mesh &spat_mesh )
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


void Field_solver::init_rhs_vector( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    //int nrow = (nx-2)*(ny-2);
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    double rhs_at_node;

    // todo: split into separate functions
    // start processing rho from the near top left corner
    for ( int k = 1; k <= nz-2; k++ ) {
	for ( int j = ny-2; j >= 1; j-- ) { 
	    for ( int i = 1; i <= nx-2; i++ ) {
		// - 4 * pi * rho * dx^2 * dy^2
		rhs_at_node = -4.0 * M_PI * spat_mesh.charge_density[i][j][k];
		rhs_at_node = rhs_at_node * dx * dx * dy * dy * dz * dz;
		// left and right boundary
		rhs_at_node = rhs_at_node
		    - dy * dy * dz * dz *
		    ( kronecker_delta(i,1) * spat_mesh.potential[0][j][k] +
		      kronecker_delta(i,nx-2) * spat_mesh.potential[nx-1][j][k] );
		// top and bottom boundary
		rhs_at_node = rhs_at_node
		    - dx * dx * dz * dz *
		    ( kronecker_delta(j,1) * spat_mesh.potential[i][0][k] +
		      kronecker_delta(j,ny-2) * spat_mesh.potential[i][ny-1][k] );
		// near and far boundary
		rhs_at_node = rhs_at_node
		    - dx * dx * dy * dy *
		    ( kronecker_delta(k,1) * spat_mesh.potential[i][j][0] +
		      kronecker_delta(k,nz-2) * spat_mesh.potential[i][j][nz-1] );
		// set rhs vector values
		// todo: separate function for:
		// (i - 1) + ( ( ny - 2 ) - j ) * (nx-2) + ( nx - 2 ) * ( ny - 2 ) * ( k - 1 )
		gsl_vector_set( rhs,
				( i - 1 ) +
				( ( ny - 2 ) - j ) * ( nx - 2 ) +
				( nx - 2 ) * ( ny - 2 ) * ( k - 1 ),
				rhs_at_node );
	    }
	}
    }    
}

int Field_solver::kronecker_delta( int i,  int j )
{
    if ( i == j ) {
	return 1;
    } else {
	return 0;
    }
}

void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    
    for( int k = 1; k <= nz-2; k++ ){
	for ( int j = ny-2; j >= 1; j-- ) { 
	    for ( int i = 1; i <= nx-2; i++ ) {
		spat_mesh.potential[i][j][k] =
		    gsl_vector_get(
			phi_vec,
			(i - 1)
			+ ( ( ny - 2 ) - j ) * (nx-2)
			+ ( nx - 2 ) * ( ny - 2 ) * ( k - 1 ) );
	    }
	}
    }
}


void Field_solver::eval_fields_from_potential( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    boost::multi_array<double, 3> &phi = spat_mesh.potential;
    double ex, ey, ez;

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int k = 0; k < nz; k++ ) {
		if ( i == 0 ) {
		    ex = - boundary_difference( phi[i][j][k], phi[i+1][j][k], dx );
		} else if ( i == nx-1 ) {
		    ex = - boundary_difference( phi[i-1][j][k], phi[i][j][k], dx );
		} else {
		    ex = - central_difference( phi[i-1][j][k], phi[i+1][j][k], dx );
		}

		if ( j == 0 ) {
		    ey = - boundary_difference( phi[i][j][k], phi[i][j+1][k], dy );
		} else if ( j == ny-1 ) {
		    ey = - boundary_difference( phi[i][j-1][k], phi[i][j][k], dy );
		} else {
		    ey = - central_difference( phi[i][j-1][k], phi[i][j+1][k], dy );
		}

		if ( k == 0 ) {
		    ez = - boundary_difference( phi[i][j][k], phi[i][j][k+1], dz );
		} else if ( k == nz-1 ) {
		    ez = - boundary_difference( phi[i][j][k-1], phi[i][j][k], dz );
		} else {
		    ez = - central_difference( phi[i][j][k-1], phi[i][j][k+1], dz );
		}

		spat_mesh.electric_field[i][j][k] = vec3d_init( ex, ey, ez );
	    }
	}
    }

    return;
}

double Field_solver::central_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

double Field_solver::boundary_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / dx );
}


Field_solver::~Field_solver()
{    
    gsl_permutation_free( pmt );
    gsl_matrix_free( a );
    gsl_vector_free( rhs );
    gsl_vector_free( phi_vec );
}
