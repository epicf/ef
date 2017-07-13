#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <iostream>
#include <mpi.h>
#include <boost/multi_array.hpp>
#include <vector>
#include "spatial_mesh.h"
#include "inner_region.h"

class Field_solver {
  public:
    std::vector<double> phi_array;
    std::vector<double> rhs_array;
    Field_solver( Spatial_mesh &spat_mesh,
		  Inner_regions_manager &inner_regions );
    void eval_potential( Spatial_mesh &spat_mesh,
			 Inner_regions_manager &inner_regions );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh );
//    virtual ~Field_solver();
  private:
    void allocate_rhs(int nrows);
    void allocate_phi_vec(int nrows);
    void solve_poisson_eqn( Spatial_mesh &spat_mesh,
			    Inner_regions_manager &inner_regions); 
    int kronecker_delta( int i,  int j );
    int node_global_index_in_matrix( Node_reference &node, int nx, int ny, int nz );
    std::vector<int> list_of_nodes_global_indices_in_matrix( std::vector<Node_reference> &nodes, int nx, int ny, int nz );
    int node_ijk_to_global_index_in_matrix( int i, int j, int k, int nx, int ny, int nz );
    void global_index_in_matrix_to_node_ijk( int global_index,
					     int *i, int *j, int *k,
					     int nx, int ny, int nz );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

#endif /* _FIELD_SOLVER_H_ */
