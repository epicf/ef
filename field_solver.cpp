#include "field_solver.h"

extern "C" void hwscrt_( double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, double *, int *, double * );

void Field_solver::eval_potential( Spatial_mesh *spat_mesh )
{
    solve_poisson_eqn( spat_mesh );
}

void Field_solver::solve_poisson_eqn( Spatial_mesh *spat_mesh )
{
    double a = 0.0;
    double b = spat_mesh->x_volume_size;
    int nx = spat_mesh->x_n_nodes;
    int M = nx-1;
    int MBDCND = 1; // 1st kind boundary conditions
    //
    double c = 0.0;
    double d = spat_mesh->y_volume_size;
    int ny = spat_mesh->y_n_nodes;
    int N = ny-1;
    int NBDCND = 1; // 1st kind boundary conditions
    //
    double BDA[ny]; // dummy
    double BDB[ny]; // dummy
    double BDC[nx]; // dummy
    double BDD[nx]; // dummy
    //
    double elmbda = 0.0;
    double **f_rhs = NULL;
    double hwscrt_f[ nx * ny ];
    int idimf = nx;
    //
    int w_dim = 
	4 * ( N + 1 ) + ( 13 + (int)( log2( N + 1 ) ) ) * ( M + 1 );
    double w[w_dim];
    double pertrb;
    int ierror;

    f_rhs = poisson_init_rhs( spat_mesh );
    rowmajor_to_colmajor( f_rhs, hwscrt_f, nx, ny );
    hwscrt_( 
	&a, &b, &M, &MBDCND, BDA, BDB,
	&c, &d, &N, &NBDCND, BDC, BDD,
	&elmbda, hwscrt_f, &idimf, &pertrb, &ierror, w);
    if ( ierror != 0 ){
	printf( "Error while solving Poisson equation (HWSCRT). \n" );
	printf( "ierror = %d \n", ierror );
    	exit( EXIT_FAILURE );
    }
    colmajor_to_rowmajor( hwscrt_f, spat_mesh->potential, nx, ny );
    poisson_free_rhs( f_rhs, nx, ny );
    return;
}


double **Field_solver::poisson_init_rhs( Spatial_mesh *spat_mesh )
{
    int nx = spat_mesh->x_n_nodes;
    int ny = spat_mesh->y_n_nodes;    

    double **rhs = NULL;
    rhs = (double **) malloc( nx * sizeof(double *) );
    if ( rhs == NULL ) {
	printf( "f_rhs allocate: nx: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	rhs[i] = (double *) malloc( ny * sizeof(double) );
	if ( rhs[i] == NULL ) {
	    printf( "f_rhs allocate: ny: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }
    
    for ( int i = 1; i < nx-1; i++ ) {
	for ( int j = 1; j < ny-1; j++ ) {
	    rhs[i][j] = -4.0 * M_PI * spat_mesh->charge_density[i][j];
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	rhs[i][0] = spat_mesh->potential[i][0];
	rhs[i][ny-1] = spat_mesh->potential[i][ny-1];
    }

    for ( int j = 0; j < ny; j++ ) {
	rhs[0][j] = spat_mesh->potential[0][j];
	rhs[nx-1][j] = spat_mesh->potential[nx-1][j];
    }
    
    return rhs;
}

void Field_solver::poisson_free_rhs( double **rhs, int nx, int ny )
{
    for( int i = 0; i < nx; i++) {
	free( rhs[i] );
    }
    free( rhs );
}

void Field_solver::rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    *( fortran + i + ( j * dim1 ) ) = c[i][j];
	}
    }
    return;
}

void Field_solver::colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    c[i][j] = *( fortran + i + ( j * dim1 ) );
	}
    }
    return;
}

void Field_solver::eval_fields_from_potential( Spatial_mesh *spat_mesh )
{
    int nx = spat_mesh->x_n_nodes;
    int ny = spat_mesh->y_n_nodes;
    double dx = spat_mesh->x_cell_size;
    double dy = spat_mesh->y_cell_size;
    double **phi = spat_mesh->potential;
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
	    spat_mesh->electric_field[i][j] = vec2d_init( ex[i][j], ey[i][j] );
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
