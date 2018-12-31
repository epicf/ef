#ifndef _FIELD_SOLVER_CUH_
#define _FIELD_SOLVER_CUH_

#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include "SpatialMeshCu.cuh"
#include "inner_region.h"

class FieldSolver {
public:
	FieldSolver(SpatialMeshCu &spat_mesh, Inner_regions_manager &inner_regions);
	void eval_potential(Inner_regions_manager &inner_regions);
	void eval_fields_from_potential();
	virtual ~FieldSolver();
private:
	SpatialMeshCu& mesh;

private:
	int max_Jacobi_iterations;
	double rel_tolerance;
	double abs_tolerance;
	double *dev_phi_next;
	//boost::multi_array<double, 3> phi_current;
	//boost::multi_array<double, 3> phi_next;
	void allocate_next_phi();
	void copy_constants_to_device();
	// Solve potential
	void solve_poisson_eqn_Jacobi(Inner_regions_manager &inner_regions);
	void single_Jacobi_iteration(Inner_regions_manager &inner_regions);
	void set_phi_next_at_boundaries();
	void compute_phi_next_at_inner_points();
	void set_phi_next_at_inner_regions(Inner_regions_manager &inner_regions);
	bool iterative_Jacobi_solutions_converged();
	void set_phi_next_as_phi_current();
};

#endif  /*_FIELD_SOLVER_CUH_*/
