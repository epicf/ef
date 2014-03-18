#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "vec2d.h"

Vec2d uniform_position_in_rectangle( const double xleft,  const double ytop,
				     const double xright, const double ybottom );
double random_in_range( const double low, const double up );

int main(int argc, char *argv[])
{
    double xleft = 4.0;
    double xright = 6.0;
    double ytop = 6.0;
    double ybottom = 4.0;
    int n = 100;
    Vec2d r;

    int seed = 0;
    srand( seed );

    for ( int i = 0; i < n; i++ ) {	
	r = uniform_position_in_rectangle( xleft, ytop, xright, ybottom );
	printf( "i: %d, ", i );
	vec2d_print(r);
	printf( "\n" );
    }

    return 0;
}


Vec2d uniform_position_in_rectangle( const double xleft,  const double ytop,
				     const double xright, const double ybottom )
{
    // Not really uniform. 
    // Will do for now.    
    return vec2d_init( random_in_range( xleft, xright ), 
		       random_in_range( ybottom, ytop ) );
}

double random_in_range( const double low, const double up )
{
    double r;
    r = ( (double)rand() / (double)RAND_MAX );
    r = low + ( up - low ) * r;
    return r;
}
