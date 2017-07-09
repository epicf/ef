#include "particle_interaction_model.h"

Particle_interaction_model::Particle_interaction_model( Config &conf ) 
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );
}

Particle_interaction_model::Particle_interaction_model(
    hid_t h5_particle_interaction_model_group )
{
    herr_t status;
    char h5_str_read_buffer[100]; // 'noninteracting' is the longest possible string
    
    status = H5LTget_attribute_string( h5_particle_interaction_model_group, "./",
				       "particle_interaction_model",
				       h5_str_read_buffer );
    hdf5_status_check( status );
    particle_interaction_model = std::string( h5_str_read_buffer );

    noninteracting = pic = false;
    if( particle_interaction_model == "noninteracting" ){
	noninteracting = true;
    } else if (	particle_interaction_model == "PIC" ){
	pic = true;
    }

    status = H5Gclose(h5_particle_interaction_model_group);
    hdf5_status_check( status );
}

void Particle_interaction_model::check_correctness_of_related_config_fields( Config &conf )
{
    std::string mode =
	conf.particle_interaction_model_config_part.particle_interaction_model;

    // 'PIC' or 'noninteracting'
    if( mode != "noninteracting" && mode != "PIC" ){
	std::cout << "Error: wrong value of 'particle_interaction_model': " + mode << std::endl;
	std::cout << "Allowed values : 'noninteracting', 'PIC'" << std::endl;
	std::cout << "Aborting" << std::endl;
	exit( EXIT_FAILURE );
    }
}

void Particle_interaction_model::get_values_from_config( Config &conf )
{
    noninteracting = pic = false;
    
    particle_interaction_model =
	conf.particle_interaction_model_config_part.particle_interaction_model;

    // todo: rewrite in civilized way. 
    if( particle_interaction_model == "noninteracting" ){
	noninteracting = true;
    } else if (	particle_interaction_model == "PIC" ){
	pic = true;
    }
}

void Particle_interaction_model::print( )
{
    std::cout << "### Particle_interaction_model:" << std::endl;
    std::cout << "Particle interaction model = "
	      << particle_interaction_model << std::endl;
    return;
}

void Particle_interaction_model::write_to_file( hid_t hdf5_file_id )
{
    hid_t group_id;
    herr_t status;
    std::string hdf5_groupname = "/Particle_interaction_model";
    group_id = H5Gcreate( hdf5_file_id, hdf5_groupname.c_str(),
			  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); hdf5_status_check( group_id );

    status = H5LTset_attribute_string( hdf5_file_id, hdf5_groupname.c_str(),
				       "particle_interaction_model",
				       particle_interaction_model.c_str() ); hdf5_status_check( status );
	
    status = H5Gclose(group_id); hdf5_status_check( status );
    return;
}

void Particle_interaction_model::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing Particle_interaction_model group. Aborting."
		  << std::endl;
	exit( EXIT_FAILURE );
    }
}
