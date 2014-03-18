#include "particles.h"

Particle particle_init( const double charge,  const double mass, 
			const Vec2d position, const Vec2d momentum );


int main(int argc, char *argv[])
{
    int n;
    Particle *p = NULL;
    particles_test_init( &p, &n );
    printf( "\n n_of_pt = %d \n", n );
    for ( int i = 0; i < n; i++ ) {
    	printf( "i:%d ", i );
    	particle_print( p[i] );
	printf( "\n" );
    }

    /* n = 100; */
    /* p = (Particle *) malloc( n_p * sizeof( Particle ) ); */
    /* if ( p == NULL ) { */
    /* 	printf( "Failed to allocate memory for particles. Aborting" ); */
    /* 	exit( EXIT_FAILURE ); */
    /* } */
    /* p[0] = particle_init( 1.0, 1.0, vec2d_init(1.0,1.0), vec2d_init(1.0,1.0) ); */
    /* particle_print( p[2] ); */


    return 0;
}
