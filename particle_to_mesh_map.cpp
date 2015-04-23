#include "particle_to_mesh_map.h"

// Eval charge density on grid
void Particle_to_mesh_map::weight_particles_charge_to_mesh( 
    Spatial_mesh &spat_mesh, Particle_sources<2> &particle_sources  )
{
    // Rewrite:
    // forall particles {
    //   find nonzero weights and corresponding nodes
    //   charge[node] = weight(particle, node) * particle.charge
    // }
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;

    for( auto& part_src: particle_sources.sources ) {
	for( auto& p : part_src.particles ) {
	    next_node_num_and_weight( p.position.x(), dx, &tr_i, &tr_x_weight );
	    next_node_num_and_weight( p.position.y(), dy, &tr_j, &tr_y_weight );
	    spat_mesh.charge_density[tr_i][tr_j] +=
		tr_x_weight * tr_y_weight * p.charge;
	    spat_mesh.charge_density[tr_i-1][tr_j] +=
		( 1.0 - tr_x_weight ) * tr_y_weight * p.charge;
	    spat_mesh.charge_density[tr_i][tr_j-1] +=
		tr_x_weight * ( 1.0 - tr_y_weight ) * p.charge;
	    spat_mesh.charge_density[tr_i-1][tr_j-1] +=
		( 1.0 - tr_x_weight ) * ( 1.0 - tr_y_weight ) * p.charge;
	}		
    }
    return;
}

VecNd<2> Particle_to_mesh_map::force_on_particle( 
    Spatial_mesh &spat_mesh, Particle<2> &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;  
    VecNd<2> field_from_node, total_field, force;
    //
    next_node_num_and_weight( p.position.x(), dx, &tr_i, &tr_x_weight );
    next_node_num_and_weight( p.position.y(), dy, &tr_j, &tr_y_weight );
    //
    total_field = VecNd<2>( true );
    field_from_node = tr_x_weight * spat_mesh.electric_field[tr_i][tr_j];
    field_from_node = tr_y_weight * field_from_node;
    total_field = total_field + field_from_node;
    //
    field_from_node = ( 1.0 - tr_x_weight ) * spat_mesh.electric_field[tr_i-1][tr_j];
    field_from_node = tr_y_weight * field_from_node;
    total_field = total_field + field_from_node;
    //
    field_from_node = tr_x_weight * spat_mesh.electric_field[tr_i][tr_j - 1];
    field_from_node = ( 1.0 - tr_y_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    //
    field_from_node = ( 1.0 - tr_x_weight ) * spat_mesh.electric_field[tr_i-1][tr_j-1];
    field_from_node = ( 1.0 - tr_y_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    //
    force = total_field * p.charge;
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
