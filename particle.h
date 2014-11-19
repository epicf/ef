#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <iostream>
#include <iomanip>
#include <deal.II/base/point.h>

template <int dim>
class Particle {
public:
    int id;
    double charge;
    double mass;
    dealii::Point<dim> position;
    dealii::Point<dim> momentum;
public:
    Particle( int id, double charge, double mass, 
	      dealii::Point<dim> position, dealii::Point<dim> momentum );
    void print();
    void print_short();
    void update_position( double dt );
    bool point_inside( const dealii::Point<dim> &p ) const;
    virtual ~Particle() {};
};

template <int dim>
Particle<dim>::Particle( int id, double charge, double mass, 
			 dealii::Point<dim> position, dealii::Point<dim> momentum ) :
    id( id ),
    charge( charge ),
    mass( mass ),
    position( position ),
    momentum( momentum )
{ }

template <int dim>
void Particle<dim>::update_position( double dt )
{
    dealii::Point<dim> pos_shift( momentum * ( dt / mass ) );
    position += pos_shift;
}

template <int dim>
bool Particle<dim>::point_inside( const dealii::Point<dim> &p ) const
{
    double particle_radius = 1.0/std::pow(2,6);
    return position.distance( p ) < particle_radius;
}

template <int dim>
void Particle<dim>::print()
{
    std::cout.setf( std::ios::fixed );
    std::cout.precision( 3 );
    std::cout << "Particle: ";
    std::cout << "id: " << id << ", ";
    std::cout << "charge = " << charge << " mass = " << mass << ", ";
    std::cout << "position = (" << position << "), ";
    std::cout << "momentum = (" << momentum << ")";
    std::cout << std::endl;
    return;
}

template <int dim>
void Particle<dim>::print_short()
{    
    std::cout.setf( std::ios::fixed );
    std::cout.precision( 2 );
    std::cout << "id: " << id << " "
	      << "position = (" << position << "), "
	      << "momentum = (" << momentum << ")"
	      << std::cout;
    return;
}


#endif /* _PARTICLE_H_ */
