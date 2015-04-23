#ifndef _EXTERNAL_MAGNETIC_FIELD_H_
#define _EXTERNAL_MAGNETIC_FIELD_H_

#include <iostream>
#include <iomanip>
#include "config.h"
#include "particle.h"
#include "vec3d.h"

class External_magnetic_field
{
public:
    Vec3d magnetic_field;
    double speed_of_light;
public:    
    External_magnetic_field( Config &conf );
    Vec3d force_on_particle( Particle &p );
    void write_to_file( std::ofstream &output_file );
    virtual ~External_magnetic_field() {};
private:
    void check_correctness_of_related_config_fields( Config &conf );
    void get_values_from_config( Config &conf );
};

#endif /* _EXTERNAL_MAGNETIC_FIELD_H_ */
