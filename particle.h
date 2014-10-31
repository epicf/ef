#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <deal.II/base/point.h>
#include <iostream>
#include <iomanip>
#include "vec2d.h"

class Particle {
public:
    int id;
    double charge;
    double mass;
    Vec2d position;
    Vec2d momentum;
public:
    Particle( int id, double charge, double mass, Vec2d position, Vec2d momentum );
    void print();
    void print_short();
    void update_position( double dt );
    bool point_inside( const dealii::Point<2> &p ) const;
    virtual ~Particle() {};
};

#endif /* _PARTICLE_H_ */
