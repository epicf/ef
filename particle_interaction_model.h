#ifndef _PARTICLE_INTERACTION_MODEL_H_
#define _PARTICLE_INTERACTION_MODEL_H_

#include <iostream>
#include <string>
#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "config.h"

class Particle_interaction_model {
  public:
    std::string particle_interaction_model;
    bool noninteracting, pic;
  public:
    Particle_interaction_model( Config &conf );
    Particle_interaction_model( hid_t h5_particle_interaction_model_group );
    void print();
    void write_to_file( hid_t hdf5_file_id );
  private:
    // initialisation
    void check_correctness_of_related_config_fields( Config &conf );
    void get_values_from_config( Config &conf );
    void check_and_exit_if_not( const bool &should_be, const std::string &message );
    // write to file
    void hdf5_status_check( herr_t status );
}; 

#endif /* _PARTICLE_INTERACTION_MODEL_H_ */
