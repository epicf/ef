#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <glib.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
    double total_time;
    double time_step_size;
    double grid_x_size;
    double grid_x_step;
    double grid_y_size;
    double grid_y_step;
    int particle_source_number_of_particles;
    double particle_source_x_left;
    double particle_source_x_right;
    double particle_source_y_bottom;
    double particle_source_y_top;
    double particle_source_temperature;
    double particle_source_charge;
    double particle_source_mass;
    double boundary_phi_left;
    double boundary_phi_right;
    double boundary_phi_bottom;
    double boundary_phi_top;
    char *output_filename_prefix;
    char *output_filename_suffix;
} Config;

void config_read( const char *config_file_name, Config *conf );
void config_check_correctness( Config *conf );
void config_print( const Config *conf );

#endif /* _CONFIG_H_ */
