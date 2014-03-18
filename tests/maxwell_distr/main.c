#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "vec2d.h"

Vec2d maxwell_momentum_distr( const double temperature, const double mass, 
			      const gsl_rng *rng);

int main(int argc, char *argv[])
{
    int n = 100;
    double temperature = 3000;
    double mass = 1.0;
    Vec2d p;

    const gsl_rng_type *rng_t = gsl_rng_default;
    gsl_rng *rng = gsl_rng_alloc( rng_t );

    for ( int i = 0; i < n; i++ ) {	
	p = maxwell_momentum_distr( temperature, mass, rng );
	printf( "i: %d, ", i);
	vec2d_print( p );
	printf( "\n" );
    }
    
    return 0;
}

Vec2d maxwell_momentum_distr( const double temperature, const double mass, 
			      const gsl_rng *rng)
{
    double maxwell_gauss_std_div = sqrt( mass * temperature * 1.0 ); // recheck
    Vec2d mom;
    mom = vec2d_init( gsl_ran_gaussian(rng, maxwell_gauss_std_div),
		      gsl_ran_gaussian(rng, maxwell_gauss_std_div) );		     
    mom = vec2d_times_scalar( mom, 1.0 ); // recheck
    return mom;
}
