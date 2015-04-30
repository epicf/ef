#include "Spatial_mesh.hpp"
#include "Particle_source.hpp"
#include "Particle.hpp"
#include "VecNd.hpp"


class Particle_to_mesh_map {
  public: 
    Particle_to_mesh_map() {};
    virtual ~Particle_to_mesh_map() {};
  public:
    void weight_particles_charge_to_mesh( 
	Spatial_mesh &spat_mesh, Particle_sources<2> &particle_sources );
    VecNd<2> force_on_particle( 
	Spatial_mesh &spat_mesh, Particle<2> &p );
  private:
    void next_node_num_and_weight( const double x, const double grid_step, 
				   int *next_node, double *weight );

};
