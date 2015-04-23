#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <iostream>
#include <iomanip>
#include "VecNd.hpp"

template< int dim >
class Particle {
  public:
    int id;
    double charge;
    double mass;
    VecNd<dim> position;
    VecNd<dim> momentum;
  public:
    Particle( int id, double charge, double mass,
	      VecNd<dim> position, VecNd<dim> momentum );
    void print();
    void print_short();
    void update_position( double dt );
    virtual ~Particle() {};
};

template< int dim >
Particle<dim>::Particle( int id, double charge, double mass,
			 VecNd<dim> position, VecNd<dim> momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum )
{ }

template< int dim >
void Particle<dim>::update_position( double dt )
{
    VecNd<dim> pos_shift;
    pos_shift = momentum * ( dt / mass );
    position = position + pos_shift;
}

template< int dim >
void Particle<dim>::print()
{
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 3 );    
    std::cout << "Particle: ";
    std::cout << "id: " << id << ", ";
    std::cout << "charge = " << charge << " mass = " << mass << ", ";
    std::cout << "position = " << position;
    std::cout << "momentum = " << momentum;
    std::cout << std::endl;
    return;
}

template< int dim >
void Particle<dim>::print_short()
{    
    std::cout.setf( std::ios::scientific );
    std::cout.precision( 2 );    
    std::cout << "id: " << id << " "
	      << "pos = " << position << " "
	      << "moment = " << momentum << " "
	      << std::endl;
    return;
}


#endif /* _PARTICLE_H_ */
