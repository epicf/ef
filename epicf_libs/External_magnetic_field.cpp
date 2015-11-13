#include "External_magnetic_field.h"

External_magnetic_field::External_magnetic_field( Config &conf )
{
    check_correctness_of_related_config_fields( conf );
    get_values_from_config( conf );    
}

void External_magnetic_field::check_correctness_of_related_config_fields( Config &conf )
{
    // nothing to check here
}

void External_magnetic_field::get_values_from_config( Config &conf )
{
    magnetic_field = vec3d_init( conf.external_magnetic_field_config_part.magnetic_field_x,
				 conf.external_magnetic_field_config_part.magnetic_field_y,
				 conf.external_magnetic_field_config_part.magnetic_field_z );
    speed_of_light = conf.external_magnetic_field_config_part.speed_of_light;
}

Vec3d External_magnetic_field::force_on_particle( Particle &p )
{
    double scale = p.charge / p.mass / speed_of_light;
    
    return vec3d_times_scalar( vec3d_cross_product( p.momentum, magnetic_field ),
			       scale );
}

void External_magnetic_field::write_to_file( std::ofstream &output_file )
{
    std::cout.precision( 3 );
    std::cout.setf( std::ios::scientific );
    std::cout.fill(' ');
    std::cout.setf( std::ios::right );
    output_file << "###External_magnetic_field" << std::endl;
    output_file << "External_magnetic_field(x,y,z) = ( "
		<< vec3d_x( magnetic_field ) << " " 
		<< vec3d_y( magnetic_field ) << " " 
		<< vec3d_z( magnetic_field ) << " )" 	
		<< std::endl;
    output_file << "Speed of light = " << speed_of_light << std::endl;
    return;
}
