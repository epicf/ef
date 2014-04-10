#include "particles.h"

Particle particle_init( const int id, const double charge, const double mass, 
			const Vec2d position, const Vec2d momentum );
Vec2d uniform_position_in_rectangle( const double xleft,  const double ytop,
				     const double xright, const double ybottom );
int generate_particle_id( const int number );
double random_in_range( const double low, const double up );
Vec2d maxwell_momentum_distr( const double temperature, const double mass, 
			      const gsl_rng *rng);


void particles_test_init( Particle **ps, int *num_of_particles )
{
    // Position
    double xleft = 4.0;
    double xright = 6.0;
    double ytop = 6.0;
    double ybottom = 4.0;
    // Momentum
    double temperature = 10;
    // Particle characteristics
    int id = 0;
    double charge = 1.0;
    double mass = 1.0;
    Vec2d pos, mom;
    *num_of_particles = 200;
    *ps = (Particle *) malloc( (*num_of_particles) * sizeof( Particle ) );
    if ( *ps == NULL ) {
	printf( "Failed to allocate memory for particles. Aborting" );
	exit( EXIT_FAILURE );
    }
        
    int seed = 0;
    srand( seed );
    
    const gsl_rng_type *rng_t = gsl_rng_default;
    gsl_rng *rng = gsl_rng_alloc( rng_t );

    for ( int i = 0; i < (*num_of_particles); i++ ) {
	id = generate_particle_id( i );
	pos = uniform_position_in_rectangle( xleft, ytop, xright, ybottom );
	mom = maxwell_momentum_distr( temperature, mass, rng );
	(*ps)[i] = particle_init( id, charge, mass, pos, mom );
    }

}

Particle particle_init( const int id, const double charge, const double mass, 
			const Vec2d position, const Vec2d momentum )
{
    Particle p;
    p.id = id;
    p.charge = charge;
    p.mass = mass;
    p.position = position;
    p.momentum = momentum;
    return p;
}

void particle_print( const Particle *p )
{
    printf( "Particle: " );
    printf( "id: %d, ", p->id );
    printf( "charge = %.3f, mass = %.3f, ", p->charge, p->mass );
    printf( "pos(x,y) = (%.3f, %.3f), ", vec2d_x(p->position), vec2d_y(p->position) );
    printf( "momentum(px,py) = (%.3f, %.3f)", vec2d_x(p->momentum), vec2d_y(p->momentum) );
    printf( "\n" );
    return;
}

void particle_print_all( const Particle *p, int n )
{
    for ( int i = 0; i < n; i++ ) {	
	particle_print( p+i );
    }
    return;
}

int generate_particle_id( const int number )
{    
    return number;
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


void particles_write_to_file( const Particle *p, const int num, FILE *f )
{
    printf( "Number of particles  = %d \n", num );
    fprintf( f, "### Particles\n" );
    fprintf( f, "Total number of particles = %d\n", num );    
    fprintf( f, "id \t   charge      mass \t\t position(x,y) \t\t momentum(px,py) \n");
    for ( int i = 0; i < num; i++ ) {	
	fprintf( f, 
		 "%-10d %-10.3f %-10.3f ( %-10.3f , %-10.3f )   ( %-10.3f , %-10.3f ) \n", 
		 p[i].id, p[i].charge, p[i].mass, 
		 vec2d_x( p[i].position ), vec2d_y( p[i].position ), 
		 vec2d_x( p[i].momentum ), vec2d_y( p[i].momentum ) );
    }
    return;
}
