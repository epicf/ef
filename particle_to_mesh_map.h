#include <mpi.h>
#include "spatial_mesh.h"
#include "particle_source.h"
#include "particle.h"
#include "vec3d.h"


class Particle_to_mesh_map {
  public: 
    Particle_to_mesh_map() {};
    virtual ~Particle_to_mesh_map() {};
  public:
    void weight_particles_charge_to_mesh( Spatial_mesh &spat_mesh,
					  Particle_sources_manager &particle_sources );
    void weight_particles_charge_to_mesh_for_single_process( Spatial_mesh &spat_mesh,
							     Particle_sources_manager &particle_sources );
    void combine_charge_densities_from_all_processes( Spatial_mesh &spat_mesh );
    Vec3d field_at_particle_position( Spatial_mesh &spat_mesh, Particle &p );
    Vec3d force_on_particle( Spatial_mesh &spat_mesh, Particle &p );
  private:
    void next_node_num_and_weight( const double x, const double grid_step, 
				   int *next_node, double *weight );

};
