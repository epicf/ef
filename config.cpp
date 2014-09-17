#include "config.h"

// Read config
void exit_if_error( GError *error );
void config_read_time_part( GKeyFile *keyfile, Config *conf, GError *error );
void config_read_spatial_mesh_part( GKeyFile *keyfile, Config *conf, GError *error );
void config_read_source_part( GKeyFile *keyfile, Config *conf, GError *error );
void config_read_boundary_conditions_part( GKeyFile *keyfile, Config *conf, GError *error );
void config_read_output_filename_part( GKeyFile *keyfile, Config *conf, GError *error );

//
// Read config
//

void config_read( const char *config_file_name, Config *conf )
{
    // Not the most elegant way.
    // Some C macroprogramming may help.
    GKeyFile *keyfile;
    GKeyFileFlags flags = G_KEY_FILE_NONE;
    GError *error = NULL;

    keyfile = g_key_file_new();
    
    g_key_file_load_from_file( keyfile, config_file_name, flags, &error );
    exit_if_error( error );

    config_read_time_part( keyfile, conf, error );
    config_read_spatial_mesh_part( keyfile, conf, error );
    config_read_source_part( keyfile, conf, error );
    config_read_boundary_conditions_part( keyfile, conf, error );
    config_read_output_filename_part( keyfile, conf, error );

    g_key_file_free( keyfile );
    return;
}

void config_read_time_part( GKeyFile *keyfile, Config *conf, GError *error )
{
    const char *time_section_name = "Time grid";
    conf->total_time = g_key_file_get_double( keyfile, time_section_name, "total_time", &error);
    exit_if_error( error );    
    conf->time_step_size = g_key_file_get_double( keyfile, time_section_name, "time_step_size", &error);
    exit_if_error( error );
    conf->time_save_step = g_key_file_get_double( keyfile, time_section_name, "time_save_step", &error);
    exit_if_error( error );
}

void config_read_spatial_mesh_part( GKeyFile *keyfile, Config *conf, GError *error )
{
    const char *spatial_mesh_section_name = "Spatial mesh";
    conf->grid_x_size = 
	g_key_file_get_double( keyfile, spatial_mesh_section_name, "grid_x_size", &error);
    exit_if_error( error );    
    conf->grid_x_step = 
	g_key_file_get_double( keyfile, spatial_mesh_section_name, "grid_x_step", &error);
    exit_if_error( error );
    conf->grid_y_size = 
	g_key_file_get_double( keyfile, spatial_mesh_section_name, "grid_y_size", &error);
    exit_if_error( error );    
    conf->grid_y_step = 
	g_key_file_get_double( keyfile, spatial_mesh_section_name, "grid_y_step", &error);
    exit_if_error( error );
}

void config_read_source_part( GKeyFile *keyfile, Config *conf, GError *error )
{
    const char *source_section_name = "Test particle source";
    conf->particle_source_number_of_particles = g_key_file_get_integer( 
	keyfile, source_section_name, "particle_source_number_of_particles", &error);
    exit_if_error( error );    
    conf->particle_source_x_left = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_x_left", &error);
    exit_if_error( error );    
    conf->particle_source_x_right = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_x_right", &error);
    exit_if_error( error );    
    conf->particle_source_y_bottom = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_y_bottom", &error);
    exit_if_error( error );    
    conf->particle_source_y_top = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_y_top", &error);
    exit_if_error( error );    
    conf->particle_source_temperature = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_temperature", &error);
    exit_if_error( error );    
    conf->particle_source_charge = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_charge", &error);
    exit_if_error( error );    
    conf->particle_source_mass = g_key_file_get_double( 
	keyfile, source_section_name, "particle_source_mass", &error);
    exit_if_error( error );    
}

void config_read_boundary_conditions_part( GKeyFile *keyfile, Config *conf, GError *error )
{
    const char *boundary_section_name = "Boundary conditions";
    conf->boundary_phi_left = g_key_file_get_double( keyfile, boundary_section_name, "boundary_phi_left", &error);
    exit_if_error( error );    
    conf->boundary_phi_right = g_key_file_get_double( keyfile, boundary_section_name, "boundary_phi_right", &error);
    exit_if_error( error );
    conf->boundary_phi_top = g_key_file_get_double( keyfile, boundary_section_name, "boundary_phi_top", &error);
    exit_if_error( error );
    conf->boundary_phi_bottom = g_key_file_get_double( keyfile, boundary_section_name, "boundary_phi_bottom", &error);
    exit_if_error( error );    
}

void config_read_output_filename_part( GKeyFile *keyfile, Config *conf, GError *error )
{
    const char *output_filename_section_name = "Output filename";
    conf->output_filename_prefix = 
	g_key_file_get_string( keyfile, output_filename_section_name, "output_filename_prefix", &error);
    conf->output_filename_suffix = 
	g_key_file_get_string( keyfile, output_filename_section_name, "output_filename_suffix", &error);
    exit_if_error( error );    
}

void exit_if_error( GError *error )
{
    if ( error ) {
	g_critical( error->message );
	exit( EXIT_FAILURE );
    }
    return;
}

//
// Print config
//

void config_print( const Config *conf )
{
    printf( "=== echo config file ===\n" );
    printf( "total_time = %f, time_step_size = %f, time_save_step = %f \n", 
	    conf->total_time, conf->time_step_size, conf->time_save_step );
    printf( "grid_x_size = %f, grid_x_step = %f, grid_y_size = %f, grid_y_step = %f \n", 
	    conf->grid_x_size, conf->grid_x_step, conf->grid_y_size, conf->grid_y_step);

    printf( "number of particles = %d \n", conf->particle_source_number_of_particles );
    printf( "source_x_left = %f, source_x_right = %f \n", conf->particle_source_x_left, conf->particle_source_x_right );
    printf( "source_y_bottom = %f, source_y_top = %f \n", conf->particle_source_y_bottom, conf->particle_source_y_top );
    printf( "source_temperature = %f \n", conf->particle_source_temperature );
    printf( "source_particle_mass = %f, source_particle_charge = %f \n",
	    conf->particle_source_charge, conf->particle_source_mass );
    printf( "boundary_phi_left = %f, boundary_phi_right = %f \n", conf->boundary_phi_left, conf->boundary_phi_right );
    printf( "boundary_phi_bottom = %f, boundary_phi_top = %f \n", conf->boundary_phi_bottom, conf->boundary_phi_top );
    printf( "output_filename_prefix = %s\n", conf->output_filename_prefix );
    printf( "output_filename_suffix = %s\n", conf->output_filename_suffix );
    printf( "=== \n" );
    printf( "\n" );
    return;
}
