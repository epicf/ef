#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void hwscrt_( double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, int *, double *, double *,
		    double *, double *, int *, double *, int *, double * );
void prepare_hwscrt_f( double **f_c, double *hwscrt_f, int nrow, int ncol );
void recover_hwscrt_f( double *hwscrt_f, double **f_c, int nrow, int ncol );
double cube( double x );  
double square( double x );

int main(int argc, char *argv[])
{
    double a = 0.0;
    double b = 10.0;
    int M = 700;
    int MBDCND = 1;
    int ncol = M+1;
    //
    double c = 0.0;
    double d = 10.0;
    int N = 700;
    int NBDCND = 1;
    int nrow = N+1;
    //
    double BDA[nrow];
    double BDB[nrow];
    double BDC[ncol];
    double BDD[ncol];
    //
    double ELMBDA = 0.0;
    int IDIMF = ncol;
    double **f_c = NULL;
    double hwscrt_f[ ncol * nrow ];
    //
    int Wdim = 4 * ( N + 1 ) + ( 13 + (int)( log2( N + 1 ) ) ) * ( M + 1 ); 
    double W[Wdim];
    //
    double PERTRB;
    int ierror;
    double x[ncol], y[nrow], Uanalit[nrow][ncol];
    double max_err = 0.0;

    f_c = (double **) malloc( nrow * sizeof(double *) );
    if ( f_c == NULL ) {
	printf( "F_C allocate: rows: out of memory ");
	exit( EXIT_FAILURE );	
    }
    for( int i = 0; i < nrow; i++) {
	f_c[i] = (double *) malloc( ncol * sizeof(double) );
	if ( f_c[i] == NULL ) {
	    printf( "F_C allocate: cols: out of memory ");
	    exit( EXIT_FAILURE );	
	}
    }

    for ( int j = 0; j < ncol; j++ ) {
	x[j] = a + (b-a) / M * j;
    }    
    for ( int i = 0; i < nrow; i++ ) {
	y[i] = c + (d-c) / N * i;
    }

    for ( int i = 0; i < nrow; i++ ) {
	for ( int j = 0; j < ncol; j++ ) {
	    Uanalit[i][j] = 
		/* cube( x[j] ) + cube( y[i] ); */
		x[j] * cos( x[j] * y[i] );
		//cube( x[j] ) * cos( x[j] * y[i] ) + 5 * x[j] * exp( -y[i] );
	}
    }

    for ( int i = 1; i < nrow - 1; i++ ) {
	for ( int j = 1; j < ncol - 1; j++ ) {
	    f_c[i][j] = 
		/* 6.0 * ( x[j] + y[i] ); */
		-2 * y[i] * sin( x[j] * y[i] ) -
		x[j] * square( y[i] ) * cos( x[j] * y[i] )
		- cube( x[j] ) * cos( x[j] * y[i] );
		/* 6 * x[j] * cos( x[j] * y[i] ) - */
		/* 6 * x[j] * x[j] * y[i] * sin( x[j] * y[i] ) -  */
		/* cube( x[j] ) * y[i] * y[i] * cos( x[j] * y[i] ) + */
		/* 5 * x[j] * exp( -y[i] ) -  */
		/* cube( x[j] ) * x[j] * x[j] * cos( x[j] * y[i] ); */
	}
    }

    for ( int j = 0; j < ncol; j++ ) {
	f_c[0][j] = Uanalit[0][j];
	f_c[nrow-1][j] = Uanalit[nrow-1][j];
    }

    for ( int i = 0; i < nrow; i++ ) {
	f_c[i][0] = Uanalit[i][0];
	f_c[i][ncol-1] = Uanalit[i][ncol-1];
    }

    /* for ( int i = 0; i < nrow; i++ ) { */
    /* 	for ( int j = 0; j < ncol; j++ ) { */
    /* 	    printf( "%.3f ", f_c[i][j] ); */
    /* 	} */
    /* 	printf( "\n" ); */
    /* }     */

    prepare_hwscrt_f( f_c, hwscrt_f, nrow, ncol );
    hwscrt_(
    	&a, &b, &M, &MBDCND, BDA, BDB,
    	&c, &d, &N, &NBDCND, BDC, BDD,
    	&ELMBDA, hwscrt_f, &IDIMF, &PERTRB, &ierror, W);
    if ( ierror != 0 ) {
    	printf( "Error while solving Poisson equation (HWSCRT): ierror = %d", ierror );
    }
    recover_hwscrt_f( hwscrt_f, f_c, nrow, ncol );
    printf( "\n" );

    for ( int i = 0; i < nrow; i++ ) {
    	for ( int j = 0; j < ncol; j++ ) {
    	    double num_an_diff = Uanalit[i][j] - f_c[i][j];
    	    if ( max_err < fabs( num_an_diff ) )
    		max_err = fabs( num_an_diff );
    	    /* printf( "%f ", f_c[i][j] ); */
    	    printf( "%.5f ", num_an_diff );
    	}
    	printf( "\n" );
    }
    printf( "pertrb = %f \n", PERTRB );
    printf( "max_an_num_diff = %f \n", max_err );

    return 0;
}


void prepare_hwscrt_f( double **f_c, double *hwscrt_f, int nrow, int ncol )
{
    for ( int i = 0; i < nrow; i++ ) {	    
	for ( int j = 0; j < ncol; j++ ) {	
	    *( hwscrt_f + i*ncol + j ) = f_c[i][j];	    
	}
    }
    return;
}

void recover_hwscrt_f( double *hwscrt_f, double **f_c, int nrow, int ncol )
{
    for ( int i = 0; i < nrow; i++ ) {	    
	for ( int j = 0; j < ncol; j++ ) {	
	    f_c[i][j] = *( hwscrt_f + i*ncol + j );
	}
    }
    return;
}

double cube( double x )
{    
    return x*x*x;
}

double square( double x )
{    
    return x*x;
}


