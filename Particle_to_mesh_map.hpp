#include "Spatial_mesh.hpp"
#include "Particle_source.hpp"
#include "Particle.hpp"
#include "VecNd.hpp"

template< int dim >
class Particle_to_mesh_map {
  public: 
    Particle_to_mesh_map() {};
    virtual ~Particle_to_mesh_map() {};
  public:
    void weight_particles_charge_to_mesh( 
	Spatial_mesh<dim> &spat_mesh, Particle_sources<dim> &particle_sources );
    VecNd<dim> force_on_particle( 
	Spatial_mesh<dim> &spat_mesh, Particle<dim> &p );
  private:
    void next_node_num_and_weight( const double x, const double grid_step, 
				   int *next_node, double *weight );

};

// Eval charge density on grid
template< int dim >
void Particle_to_mesh_map<dim>::weight_particles_charge_to_mesh( 
    Spatial_mesh<dim> &spat_mesh, Particle_sources<dim> &particle_sources  )
{   
    for( auto& part_src: particle_sources.sources ) {
	for( auto& p : part_src.particles ) {
	    contribute_charge( spat_mesh, p );
	}		
    }
    return;
}

template< int dim >
void Particle_to_mesh_map<dim>::contribute_charge( 
    Spatial_mesh<dim> &spat_mesh, Particle<dim> &p  )
{
    std::cout << "Unsupported dim=" << dim << " in Particle_to_mesh_map. Aborting.";
    exit( EXIT_FAILURE );
    return;
}

// Rewrite: for each particle determine nodes it contributes to
// and an amount of the contribution. Return array( std::vectors? ) of
// nodes and contributions.
// Separate routine to perform contribution.

template<>
void Particle_to_mesh_map<1>::contribute_charge( 
    Spatial_mesh<1> &spat_mesh, Particle<1> &p  )
{
    double dx = spat_mesh.x_cell_size;
    int tr_i; // 'tr' = 'top_right'
    double tr_x_weight;

    next_node_num_and_weight( p.position.x(), dx, &tr_i, &tr_x_weight );
    spat_mesh.charge_density[tr_i] +=
	tr_x_weight * p.charge;
    spat_mesh.charge_density[tr_i-1] +=
	( 1.0 - tr_x_weight ) * p.charge;

    return;
}

template<>
void Particle_to_mesh_map<2>::contribute_charge( 
    Spatial_mesh<2> &spat_mesh, Particle<2> &p  )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;

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

    return;
}


template<>
void Particle_to_mesh_map<3>::contribute_charge( 
    Spatial_mesh<3> &spat_mesh, Particle<3> &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int trf_i, trf_j, trf_k; // 'trf' = 'top_right_far'
    double trf_x_weight, trf_y_weight, trf_z_weight;

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

    return;
}



template< int dim >
VecNd<dim> Particle_to_mesh_map<dim>::force_on_particle( 
    Spatial_mesh<dim> &spat_mesh, Particle<dim> &p )
{
    std::cout << "Unsupported dim=" << dim << " in Particle_to_mesh_map. Aborting.";
    exit( EXIT_FAILURE );
    return;
}

template<>
VecNd<1> Particle_to_mesh_map<1>::force_on_particle( 
    Spatial_mesh<1> &spat_mesh, Particle<1> &p )
{
    double dx = spat_mesh.x_cell_size;
    int tr_i; // 'tr' = 'top_right'
    double tr_x_weight;  
    VecNd<dim> field_from_node, total_field, force;
    //
    next_node_num_and_weight( p.position.x(), dx, &tr_i, &tr_x_weight );
    //
    total_field = VecNd<1>( true );
    field_from_node = tr_x_weight * spat_mesh.electric_field[tr_i];
    total_field = total_field + field_from_node;
    //
    field_from_node = ( 1.0 - tr_x_weight ) * spat_mesh.electric_field[tr_i-1];
    total_field = total_field + field_from_node;
    //
    force = total_field * p.charge;
    return force;
}

template<>
VecNd<2> Particle_to_mesh_map<2>::force_on_particle( 
    Spatial_mesh<2> &spat_mesh, Particle<2> &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    int tr_i, tr_j; // 'tr' = 'top_right'
    double tr_x_weight, tr_y_weight;  
    VecNd<dim> field_from_node, total_field, force;
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

template<>
VecNd<3> Particle_to_mesh_map<3>::force_on_particle( 
    Spatial_mesh<3> &spat_mesh, Particle<3> &p )
{
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    int trf_i, trf_j, trf_k; // 'trf' = 'top_right_far'
    double trf_x_weight, trf_y_weight, trf_z_weight;  
    Vec3d field_from_node, total_field, force;
    //
    next_node_num_and_weight( p.position.x(), dx, &trf_i, &trf_x_weight );
    next_node_num_and_weight( p.position.y(), dy, &trf_j, &trf_y_weight );
    next_node_num_and_weight( p.position.z(), dz, &trf_k, &trf_z_weight );
    // trf
    total_field = VecNd<3>( true );
    field_from_node = trf_x_weight * spat_mesh.electric_field[trf_i][trf_j][trf_k];
    field_from_node = trf_y_weight * field_from_node;
    field_from_node = trf_z_weight * field_from_node;
    total_field = total_field + field_from_node;
    // tlf
    field_from_node = ( 1.0 - trf_x_weight ) * spat_mesh.electric_field[trf_i-1][trf_j][trf_k];
    field_from_node = trf_y_weight * field_from_node;
    field_from_node = trf_z_weight * field_from_node;
    total_field = total_field + field_from_node;
    // brf
    field_from_node = trf_x_weight * spat_mesh.electric_field[trf_i][trf_j - 1][trf_k];
    field_from_node = ( 1.0 - trf_y_weight ) * field_from_node;
    field_from_node = trf_z_weight * field_from_node;
    total_field = total_field + field_from_node;
    // blf
    field_from_node = ( 1.0 - trf_x_weight ) * spat_mesh.electric_field[trf_i-1][trf_j-1][trf_k];
    field_from_node = ( 1.0 - trf_y_weight ) * field_from_node;
    field_from_node = trf_z_weight * field_from_node;
    total_field = total_field + field_from_node;
    // trn
    field_from_node = trf_x_weight * spat_mesh.electric_field[trf_i][trf_j][trf_k-1];
    field_from_node = trf_y_weight * field_from_node;
    field_from_node = ( 1.0 - trf_z_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    // tln
    field_from_node = ( 1.0 - trf_x_weight ) * spat_mesh.electric_field[trf_i-1][trf_j][trf_k-1];
    field_from_node = trf_y_weight * ield_from_node;
    field_from_node = ( 1.0 - trf_z_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    // brn
    field_from_node = trf_x_weight * spat_mesh.electric_field[trf_i][trf_j - 1][trf_k-1];
    field_from_node = ( 1.0 - trf_y_weight ) * field_from_node;
    field_from_node = ( 1.0 - trf_z_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    // bln
    field_from_node = ( 1.0 - trf_x_weight ) * spat_mesh.electric_field[trf_i-1][trf_j-1][trf_k-1];
    field_from_node = ( 1.0 - trf_y_weight ) * field_from_node;
    field_from_node = ( 1.0 - trf_z_weight ) * field_from_node;
    total_field = total_field + field_from_node;
    //
    force = total_field * p.charge;
    return force;
}

void Particle_to_mesh_map<dim>::next_node_num_and_weight( 
    const double x, const double grid_step, 
    int *next_node, double *weight )
{
    double x_in_grid_units = x / grid_step;
    *next_node = ceil( x_in_grid_units );
    *weight = 1.0 - ( *next_node - x_in_grid_units );
    return;
}
