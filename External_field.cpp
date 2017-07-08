#include "External_field.h"

External_field::External_field( External_field_config_part &field_conf )
{
    name = field_conf.name;
}

External_field::External_field( hid_t h5_external_field_group )
{
    herr_t status;

    size_t grp_name_size = 0;
    char *grp_name = NULL;
    grp_name_size = H5Iget_name( h5_external_field_group, grp_name, grp_name_size );
    grp_name_size = grp_name_size + 1;
    grp_name = new char[ grp_name_size ];
    grp_name_size = H5Iget_name( h5_external_field_group, grp_name, grp_name_size );
    std::string longname = std::string( grp_name );
    name = longname.substr( longname.find_last_of("/") + 1 );
    delete[] grp_name;
}

void External_field::write_to_file( hid_t fields_group_id )
{
    hid_t current_field_group_id;
    herr_t status;
    std::string hdf5_groupname = "./" + name;
    std::string current_group = "./";

    current_field_group_id = H5Gcreate( fields_group_id, hdf5_groupname.c_str(),
					H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5_status_check( current_field_group_id );


    write_hdf5_field_parameters( current_field_group_id );

    
    status = H5Gclose( current_field_group_id ); hdf5_status_check( status );
    return;
}

void External_field::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing External_field "
		  << name << "."
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}


// Uniform magnetic

External_field_uniform_magnetic::External_field_uniform_magnetic(
    External_field_uniform_magnetic_config_part &field_conf ) :
    External_field( field_conf )
{
    field_type = "uniform_magnetic";
    check_correctness_of_related_config_fields( field_conf );
    get_values_from_config( field_conf );
}

void External_field_uniform_magnetic::check_correctness_of_related_config_fields(
    External_field_uniform_magnetic_config_part &field_conf )
{
    // nothing to check here
}

void External_field_uniform_magnetic::get_values_from_config(
    External_field_uniform_magnetic_config_part &field_conf )
{
    magnetic_field = vec3d_init( field_conf.magnetic_field_x,
				 field_conf.magnetic_field_y,
				 field_conf.magnetic_field_z );
    speed_of_light = field_conf.speed_of_light;
}

External_field_uniform_magnetic::External_field_uniform_magnetic(
    hid_t h5_external_field_uniform_magnetic_group ) :
    External_field( h5_external_field_uniform_magnetic_group )
{
    herr_t status;
    double H_x, H_y, H_z;

    field_type = "uniform_magnetic";
    
    status = H5LTget_attribute_double( h5_external_field_uniform_magnetic_group, "./",
				       "uniform_magnetic_field_x", &H_x );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_field_uniform_magnetic_group, "./",
				       "uniform_magnetic_field_y", &H_y );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_field_uniform_magnetic_group, "./",
				       "uniform_magnetic_field_z", &H_z );
    hdf5_status_check( status );
    status = H5LTget_attribute_double( h5_external_field_uniform_magnetic_group, "./",
				       "speed_of_light", &speed_of_light );
    hdf5_status_check( status );

    magnetic_field = vec3d_init( H_x, H_y, H_z );
}

Vec3d External_field_uniform_magnetic::force_on_particle( Particle &p )
{
    double scale = p.charge / p.mass / speed_of_light;
    
    return vec3d_times_scalar( vec3d_cross_product( p.momentum, magnetic_field ),
			       scale );
}

void External_field_uniform_magnetic::write_hdf5_field_parameters(
    hid_t current_field_group_id )
{
    double H_x = vec3d_x( magnetic_field );
    double H_y = vec3d_y( magnetic_field );
    double H_z = vec3d_z( magnetic_field );

    herr_t status;
    int single_element = 1;
    std::string current_group = "./";

    status = H5LTset_attribute_string( current_field_group_id, current_group.c_str(),
				       "field_type", field_type.c_str() );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "uniform_magnetic_field_x", &H_x, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "uniform_magnetic_field_y", &H_y, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "uniform_magnetic_field_z", &H_z, single_element );
    hdf5_status_check( status );
    status = H5LTset_attribute_double( current_field_group_id, current_group.c_str(),
				       "speed_of_light", &speed_of_light, single_element );
    hdf5_status_check( status );
    
    return;
}

void External_field_uniform_magnetic::hdf5_status_check( herr_t status )
{
    if( status < 0 ){
	std::cout << "Something went wrong while reading or writing "
		  << "External_field_uniform_magnetic group. "
		  << "Aborting." << std::endl;
	exit( EXIT_FAILURE );
    }
}
