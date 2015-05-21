#include "particle.h"

Particle::Particle( int id, double charge, double mass, Vec2d position, Vec2d momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum )
{ }


void Particle::update_position( double dt )
{
    Vec2d pos_shift;            
    pos_shift = vec2d_times_scalar( momentum, dt / mass );
    position = vec2d_add( position, pos_shift );
}

void Particle::print()
{
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 3 );    
    std::cout << "Particle: ";
    std::cout << "id: " << id << ", ";
    std::cout << "charge = " << charge << " mass = " << mass << ", ";
    std::cout << "pos(x,y) = (" << vec2d_x( position ) << ", " << vec2d_y( position ) << "), ";
    std::cout << "momentum(px,py) = (" << vec2d_x( momentum ) << ", " << vec2d_y( momentum ) << ")";
    std::cout << std::endl;
    return;
}


void Particle::print_short()
{    
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 2 );    
    std::cout << "id: " << id << " "
	      << "x = " << vec2d_x( position ) << " "
	      << "y = " << vec2d_y( position ) << " "
	      << "px = " << vec2d_x( momentum ) << " "
	      << "py = " <<  vec2d_y( momentum ) << " "
	      << std::cout;
    return;
}
