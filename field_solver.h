#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <iostream>
#include <petscksp.h>
#include <mpi.h>
#include <boost/multi_array.hpp>
#include <vector>
#include "spatial_mesh.h"
#include "inner_region.h"

class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh,
		  Inner_regions_manager &inner_regions );
    void eval_potential( Spatial_mesh &spat_mesh,
			 Inner_regions_manager &inner_regions );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh );
    virtual ~Field_solver();
  private:
    Vec phi_vec, rhs;
    Mat A;
    KSP ksp;
    PC pc;
    PetscInt rstart, rend, nlocal;
    void alloc_petsc_vector( Vec *x, PetscInt size, const char *name );
    void get_vector_ownership_range_and_local_size_for_each_process(
	Vec *x, PetscInt *rstart, PetscInt *rend, PetscInt *nlocal );
    void alloc_petsc_matrix( Mat *A,
			     PetscInt nrow_local, PetscInt ncol_local,
			     PetscInt nrow, PetscInt ncol,
			     PetscInt nonzero_per_row );
    void alloc_petsc_matrix_seqaij( Mat *A, PetscInt nrow, PetscInt ncol, PetscInt nonzero_per_row );
    void construct_equation_matrix( Mat *A,
				    Spatial_mesh &spat_mesh,				    
				    Inner_regions_manager &inner_regions,
				    PetscInt nlocal, PetscInt rstart, PetscInt rend );
    void construct_equation_matrix_in_full_domain( Mat *A,
						   int nx, int ny, int nz,
						   double dx, double dy, double dz,
						   PetscInt nlocal, PetscInt rstart, PetscInt rend );
    void cross_out_nodes_occupied_by_objects( Mat *A,
					      int nx, int ny, int nz,
					      Inner_regions_manager &inner_regions ); 
    void cross_out_nodes_occupied_by_objects( Mat *A,
					      int nx, int ny, int nz,
					      Inner_region &inner_region );
    void modify_equation_near_object_boundaries( Mat *A,
						 int nx, int ny, int nz,
						 double dx, double dy, double dz,
						 Inner_regions_manager &inner_regions ); 
    void modify_equation_near_object_boundaries( Mat *A,
						 int nx, int ny, int nz,
						 double dx, double dy, double dz,
						 Inner_region &inner_region );

    std::vector<PetscInt> adjacent_nodes_not_at_domain_edge_and_inside_inner_region(
	Node_reference &node,
	Inner_region &inner_region,
	int nx, int ny, int nz,
	double dx, double dy, double dz );
    void create_solver_and_preconditioner( KSP *ksp, PC *pc, Mat *A );
    void construct_d2dx2_in_3d( Mat *d2dx2_3d, int nx, int ny, int nz, PetscInt rstart, PetscInt rend );
    void construct_d2dy2_in_3d( Mat *d2dy2_3d, int nx, int ny, int nz, PetscInt rstart, PetscInt rend );
    void construct_d2dz2_in_3d( Mat *d2dz2_3d, int nx, int ny, int nz, PetscInt rstart, PetscInt rend );
    // Solve potential
    void solve_poisson_eqn( Spatial_mesh &spat_mesh,
			    Inner_regions_manager &inner_regions ); 
    void init_rhs_vector( Spatial_mesh &spat_mesh,
			  Inner_regions_manager &inner_regions ); 
    void init_rhs_vector_in_full_domain( Spatial_mesh &spat_mesh );
    void set_rhs_at_nodes_occupied_by_objects( Spatial_mesh &spat_mesh,
					       Inner_regions_manager &inner_regions ); 
    void set_rhs_at_nodes_occupied_by_objects( Spatial_mesh &spat_mesh,
					       Inner_region &inner_region );
    void modify_rhs_near_object_boundaries( Spatial_mesh &spat_mesh,
					    Inner_regions_manager &inner_regions ); 
    void modify_rhs_near_object_boundaries( Spatial_mesh &spat_mesh,
					    Inner_region &inner_region );
    void indicies_of_near_boundary_nodes_and_rhs_modifications(
	std::vector<PetscInt> &indices_of_nodes_near_boundaries,
	std::vector<PetscScalar> &rhs_modification_for_nodes_near_boundaries,
	int nx, int ny, int nz,
	double dx, double dy, double dz,
	Inner_region &inner_region );
    void set_solution_at_nodes_of_inner_regions( Spatial_mesh &spat_mesh,
						 Inner_regions_manager &inner_regions ); 
    void set_solution_at_nodes_of_inner_regions( Spatial_mesh &spat_mesh,
						 Inner_region &inner_region );
    int kronecker_delta( int i,  int j );
    int node_global_index_in_matrix( Node_reference &node, int nx, int ny, int nz );
    std::vector<int> list_of_nodes_global_indices_in_matrix( std::vector<Node_reference> &nodes, int nx, int ny, int nz );
    int node_ijk_to_global_index_in_matrix( int i, int j, int k, int nx, int ny, int nz );
    void global_index_in_matrix_to_node_ijk( int global_index,
					     int *i, int *j, int *k,
					     int nx, int ny, int nz );
    void transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh );
    void bcast_phi_array_size( int *recieved_rstart, int *recieved_rend, int *recieved_nlocal,
			       int proc, int mpi_process_rank );
    void allocate_and_populate_phi_array( double **local_phi_values, int recieved_nlocal,
					  int proc, int mpi_process_rank );
    void transfer_from_phi_array_to_spat_mesh_potential( double *local_phi_values,
							 int recieved_rstart, int recieved_rend,
							 Spatial_mesh &spat_mesh );
    void deallocate_phi_array( double *local_phi_values, int proc, int mpi_process_rank );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

#endif /* _FIELD_SOLVER_H_ */
