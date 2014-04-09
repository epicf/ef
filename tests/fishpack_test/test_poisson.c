#include <stdio.h>
#include <math.h>
#include <stdlib.h>

extern void hwscrt_( double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, double *, int *, double * );
void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 );
void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 );		  
void poisson_free_rhs( double **rhs, int nrow, int ncol );
double cube( double x );  

int main(int argc, char *argv[])
{
    double a = 0.0;
    double b = 10.0;
    int nx = 30;
    int M = nx-1;
    int MBDCND = 1; // 1st kind boundary conditions
    //
    double c = 0.0;
    double d = 10.0;
    int ny = 50;
    int N = ny-1;
    int NBDCND = 1; // 1st kind boundary conditions
    //
    double BDA[ny]; // dummy
    double BDB[ny]; // dummy
    double BDC[nx]; // dummy
    double BDD[nx]; // dummy
    //
    double elmbda = 0.0;
    double **rhs = NULL;
    double hwscrt_rhs[ nx * ny ];
    int idimf = nx;
    //
    int w_dim = 
	4 * ( N + 1 ) + ( 13 + (int)( log2( N + 1 ) ) ) * ( M + 1 );
    double w[w_dim];
    double pertrb;
    int ierror;
    //
    double x[nx], y[ny], Uanalit[nx][ny];

    for ( int i = 0; i < nx; i++ ) {
	x[i] = a + (b-a) / M * i;
    }

    for ( int j = 0; j < ny; j++ ) {
	y[j] = c + (d-c) / N * j;
    }

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    //Uanalit[i][j] = ( cube( x[i] ) + cube( y[j] ) ) / 6;
	    Uanalit[i][j] = ( cube( x[i] ) * cube( y[j] ) ) / 6;
	}
    }

    rhs = (double **) malloc( nx * sizeof(double *) );
    if ( rhs == NULL ) {
	printf( "rhs allocate: nx: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nx; i++) {
	rhs[i] = (double *) malloc( ny * sizeof(double) );
	if ( rhs[i] == NULL ) {
	    printf( "rhs allocate: ny: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }

    for ( int i = 1; i < nx-1; i++ ) {
	for ( int j = 1; j < ny-1; j++ ) {
	    //rhs[i][j] = x[i] + y[j];
	    rhs[i][j] = x[i] * cube( y[j] ) + cube( x[i] ) * y[j];
	}
    }

    for ( int i = 0; i < nx; i++ ) {
	rhs[i][0] = Uanalit[i][0];
	rhs[i][ny-1] = Uanalit[i][ny-1];
    }

    for ( int j = 0; j < ny; j++ ) {
	rhs[0][j] = Uanalit[0][j];
	rhs[nx-1][j] = Uanalit[nx-1][j];
    }

    rowmajor_to_colmajor( rhs, hwscrt_rhs, nx, ny );
    hwscrt_( 
	&a, &b, &M, &MBDCND, BDA, BDB,
	&c, &d, &N, &NBDCND, BDC, BDD,
	&elmbda, hwscrt_rhs, &idimf, &pertrb, &ierror, w);
    if ( ierror != 0 ) {
	printf( "Error while solving Poisson equation (HWSCRT): ierror = %d", ierror );
    }
    colmajor_to_rowmajor( hwscrt_rhs, rhs, nx, ny );
    printf( "pertrb = %f \n", pertrb );

    printf( "Unalit and num difference\n" );
    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    //printf( "%f ", rhs[i][j] );
	    printf( "%f ", Uanalit[i][j] - rhs[i][j] );
	}
	printf( "\n" );
    }

    poisson_free_rhs( rhs, nx, ny );
    return 0;
}

void poisson_free_rhs( double **rhs, int nx, int ny )
{
    for( int i = 0; i < nx; i++) {
	free( rhs[i] );
    }
    free( rhs );
}

void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    *( fortran + i + ( j * dim1 ) ) = c[i][j];
	}
    }
    return;
}

void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 )
{
    for ( int j = 0; j < dim2; j++ ) {
	for ( int i = 0; i < dim1; i++ ) {
	    c[i][j] = *( fortran + i + ( j * dim1 ) );
	}
    }
    return;
}

double cube( double x )
{    
    return x*x*x;
}


