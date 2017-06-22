#ifndef _EXTERNAL_MAGNETIC_FIELD_H_
#define _EXTERNAL_MAGNETIC_FIELD_H_

#include <iostream>
#include <iomanip>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <mpi.h>
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
    External_magnetic_field( hid_t h5_external_magnetic_field_group );
    Vec3d force_on_particle( Particle &p );
    void write_to_file( hid_t hdf5_file_id );
    virtual ~External_magnetic_field() {};
private:
    void check_correctness_of_related_config_fields( Config &conf );
    void get_values_from_config( Config &conf );
    void hdf5_status_check( herr_t status );
};

#endif /* _EXTERNAL_MAGNETIC_FIELD_H_ */
