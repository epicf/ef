#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include "iostream"
#include "iomanip"
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
    virtual ~Particle() {};
};

#endif /* _PARTICLE_H_ */
