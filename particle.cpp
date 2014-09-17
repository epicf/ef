#include "particle.h"

Particle::Particle( int id, double charge, double mass, Vec2d position, Vec2d momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum )
{ }


void Particle::print()
{
    printf( "Particle: " );
    printf( "id: %d, ", id );
    printf( "charge = %.3f, mass = %.3f, ", charge, mass );
    printf( "pos(x,y) = (%.3f, %.3f), ", vec2d_x( position ), vec2d_y( position ) );
    printf( "momentum(px,py) = (%.3f, %.3f)", vec2d_x( momentum ), vec2d_y( momentum ) );
    printf( "\n" );
    return;
}
