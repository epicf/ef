#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <cstdio>
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
    virtual ~Particle() {};
};

#endif /* _PARTICLE_H_ */
