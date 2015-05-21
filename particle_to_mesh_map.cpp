#include "particle_to_mesh_map.h"

// Eval charge density on grid
void Particle_to_mesh_map::weight_particles_charge_to_mesh( 
    Spatial_mesh &spat_mesh, Particle_sources &particle_sources  )
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   charge[node] = weight(particle, node) * particle.charge
    // }
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int trf_i, trf_j, trf_k; // 'trf' = 'top_right_far'
    double trf_x_weight, trf_y_weight, trf_z_weight;

    for( auto& part_src: particle_sources.sources ) {
	for( auto& p : part_src.particles ) {
	    next_node_num_and_weight( vec3d_x( p.position ), dx, &trf_i, &trf_x_weight );
	    next_node_num_and_weight( vec3d_y( p.position ), dy, &trf_j, &trf_y_weight );
	    next_node_num_and_weight( vec3d_z( p.position ), dz, &trf_k, &trf_z_weight );
	    spat_mesh.charge_density[trf_i][trf_j][trf_k] +=
		trf_x_weight * trf_y_weight * trf_z_weight * p.charge;
	    spat_mesh.charge_density[trf_i-1][trf_j][trf_k] +=
		( 1.0 - trf_x_weight ) * trf_y_weight * trf_z_weight * p.charge;
	    spat_mesh.charge_density[trf_i][trf_j-1][trf_k] +=
		trf_x_weight * ( 1.0 - trf_y_weight ) * trf_z_weight * p.charge;
	    spat_mesh.charge_density[trf_i-1][trf_j-1][trf_k] +=
		( 1.0 - trf_x_weight ) * ( 1.0 - trf_y_weight ) * trf_z_weight * p.charge;
	    spat_mesh.charge_density[trf_i][trf_j][trf_k - 1] +=
		trf_x_weight * trf_y_weight * ( 1.0 - trf_z_weight ) * p.charge;
	    spat_mesh.charge_density[trf_i-1][trf_j][trf_k - 1] +=
		( 1.0 - trf_x_weight ) * trf_y_weight * ( 1.0 - trf_z_weight ) * p.charge;
	    spat_mesh.charge_density[trf_i][trf_j-1][trf_k - 1] +=
		trf_x_weight * ( 1.0 - trf_y_weight ) * ( 1.0 - trf_z_weight ) * p.charge;
	    spat_mesh.charge_density[trf_i-1][trf_j-1][trf_k - 1] +=
		( 1.0 - trf_x_weight ) * ( 1.0 - trf_y_weight ) * ( 1.0 - trf_z_weight ) * p.charge;

	}		
    }
    return;
}

Vec3d Particle_to_mesh_map::force_on_particle( 
    Spatial_mesh &spat_mesh, Particle &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int trf_i, trf_j, trf_k; // 'trf' = 'top_right_far'
    double trf_x_weight, trf_y_weight, trf_z_weight;  
    Vec3d field_from_node, total_field, force;
    //
    next_node_num_and_weight( vec3d_x( p.position ), dx, &trf_i, &trf_x_weight );
    next_node_num_and_weight( vec3d_y( p.position ), dy, &trf_j, &trf_y_weight );
    next_node_num_and_weight( vec3d_z( p.position ), dz, &trf_k, &trf_z_weight );
    // trf
    total_field = vec3d_zero();
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i][trf_j][trf_k],
	trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tlf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i-1][trf_j][trf_k],
	1.0 - trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brf
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i][trf_j - 1][trf_k],	
	trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // blf
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[trf_i-1][trf_j-1][trf_k],	
	1.0 - trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // trn
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i][trf_j][trf_k-1],
	trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // tln
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i-1][trf_j][trf_k-1],
	1.0 - trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // brn
    field_from_node = vec3d_times_scalar(
	spat_mesh.electric_field[trf_i][trf_j - 1][trf_k-1],	
	trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );
    // bln
    field_from_node = vec3d_times_scalar(			
	spat_mesh.electric_field[trf_i-1][trf_j-1][trf_k-1],	
	1.0 - trf_x_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_y_weight );
    field_from_node = vec3d_times_scalar( field_from_node, 1.0 - trf_z_weight );
    total_field = vec3d_add( total_field, field_from_node );    
    //
    force = vec3d_times_scalar( total_field, p.charge );
    return force;
}

void Particle_to_mesh_map::next_node_num_and_weight( 
    const double x, const double grid_step, 
    int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}
