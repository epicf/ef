#include "particle.h"

Particle::Particle( int id, double charge, double mass, Vec3d position, Vec3d momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum ),
    momentum_is_half_time_step_shifted( false )
{ }


void Particle::update_position( double dt )
{
    Vec3d pos_shift;            
    pos_shift = vec3d_times_scalar( momentum, dt / mass );
    position = vec3d_add( position, pos_shift );
}

void Particle::print()
{
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 3 );    
    std::cout << "Particle: ";
    std::cout << "id: " << id << ", ";
    std::cout << "charge = " << charge << " mass = " << mass << ", ";
    std::cout << "pos(x,y,z) = ("
	      << vec3d_x( position ) << ", "
	      << vec3d_y( position ) << ", "
	      << vec3d_z( position ) << "), ";
    std::cout << "momentum(px,py,pz) = ("
	      << vec3d_x( momentum ) << ", "
	      << vec3d_y( momentum ) << ", "
	      << vec3d_z( momentum ) << ")";
    std::cout << std::endl;
    return;
}


void Particle::print_short()
{    
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 2 );    
    std::cout << "id: " << id << " "
	      << "x = " << vec3d_x( position ) << " "
	      << "y = " << vec3d_y( position ) << " "
	      << "z = " << vec3d_z( position ) << " "	
	      << "px = " << vec3d_x( momentum ) << " "
	      << "py = " <<  vec3d_y( momentum ) << " "
	      << "pz = " <<  vec3d_z( momentum ) << " "
	      << std::endl;
    return;
}
