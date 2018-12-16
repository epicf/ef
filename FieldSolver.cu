#include "FieldSolver.cuh"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "math_constants.h"

#define ABS_TOLERANCE = 1.0e-5;
#define REL_TOLERANCE = 1.0e-12;

__constant__ double dxdxdydy[1];
__constant__ double dxdxdzdz[1];
__constant__ double dydydzdz[1];
__constant__ double dxdxdydydzdz[1];

__constant__ int end[1];

__device__ int GetIdxVolume_NoBorder() {
	//int xStepthread = 1;
	int xStepBlock = blockDim.x;
	int yStepThread = d_n_nodes[0].x;
	int yStepBlock = yStepThread * blockDim.y;
	int zStepThread = d_n_nodes[0].x * d_n_nodes[0].y;
	int zStepBlock = zStepThread * blockDim.z;
	return (threadIdx.x + blockIdx.x * xStepBlock) +
		(threadIdx.y * yStepThread + blockIdx.y * yStepBlock) +
		(threadIdx.z * zStepThread + blockIdx.z * zStepBlock);
}

__device__ double GradientComponent(double phi1, double phi2, double cell_side_size) {
	return ((phi2 - phi1) / cell_side_size);
}

__global__ void SetPhiNextAsCurrent(double* d_phi_current, double* d_phi_next) {
	int idx = GetIdxVolume_NoBorder();
	d_phi_current[idx] = d_phi_next[idx];
}

__global__ void ComputePhiNext(const double* d_phi_current, const double* d_charge, double* d_phi_next) {
	int idx = GetIdxVolume_NoBorder();
	int offset_Dx = 1;
	//todo rewrite usind device n_nodes.x/y/z
	int offset_Dy = blockDim.x * gridDim.x;
	int offset_Dz = offset_Dy * blockDim.y * gridDim.y;

	int prev_neibhour_idx;
	int next_neibhour_idx;

	//double dxdxdydy = mesh.volume_size.x * mesh.volume_size.x *
	//	mesh.volume_size.y * mesh.volume_size.y;
	//double dxdxdzdz = mesh.volume_size.x * mesh.volume_size.x *
	//	mesh.volume_size.z * mesh.volume_size.z;
	//double dydydzdz = mesh.volume_size.y * mesh.volume_size.y *
	//	mesh.volume_size.z * mesh.volume_size.z;

	//double dxdxdydydzdz = mesh.volume_size.x * mesh.volume_size.x *
	//	dy * dy * dz * dz;
	double denom = (double)2* (dxdxdydy[0] + dxdxdzdz[0] + dydydzdz[0]);
	////
	prev_neibhour_idx = max(idx + offset_Dx,0);
	next_neibhour_idx = min(idx + offset_Dx,end[0]);//dirty : can be optimized for configs where n_nodes side equals (k*POT+2)
	d_phi_next[idx] =
		(d_phi_current[next_neibhour_idx] + d_phi_current[prev_neibhour_idx])*dydydzdz[0];

	prev_neibhour_idx = max(idx + offset_Dy, 0);
	next_neibhour_idx = min(idx + offset_Dy, end[0]);
	d_phi_next[idx] +=
		(d_phi_current[next_neibhour_idx] + d_phi_current[prev_neibhour_idx])*dxdxdzdz[0];

	prev_neibhour_idx = max(idx + offset_Dz, 0);
	next_neibhour_idx = min(idx + offset_Dz, end[0]);
	d_phi_next[idx] +=
		(d_phi_current[next_neibhour_idx] + d_phi_current[prev_neibhour_idx])*dxdxdydy[0];

	d_phi_next[idx] +=	4.0 * CUDART_PI * d_charge[idx] * dxdxdydydzdz[0];
	d_phi_next[idx] /= denom;

}

__global__ void EvaluateFields(const double* dev_potential, double3* dev_el_field) {
	int idx = GetIdxVolume_NoBorder();

	double3 e = make_double3(0, 0, 0);
	bool is_on_up_border;
	bool is_on_low_border;
	bool is_inside_borders;
	int offset;

	offset = 1;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.x = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		dev_cell_size.x);

	offset = d_n_nodex.x;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.y = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		dev_cell_size.y);

	offset = d_n_nodes.y*d_n_nodes.x;
	is_on_up_border = ((threadIdx.x == 0) && (blockIdx.x == 0));
	is_on_low_border = ((threadIdx.x == (blockDim.x - 1)) && (blockIdx.x == (gridDim.x - 1)));
	is_inside_borders = !(is_on_low_border || is_on_up_border);

	e.z = -(1 / (1 + is_inside_borders)) * GradientComponent(
		dev_potential[idx + (offset*is_on_up_border) - (offset*is_inside_borders)],
		dev_potential[idx - (offset*is_on_low_border) + (offset*is_inside_borders)],
		dev_cell_size.z);

	dev_el_field[idx] = e;

}

FieldSolver::FieldSolver(SpatialMeshCu &mesh, Inner_regions_manager &inner_regions) :mesh(mesh)
{
	allocate_next_phi();
}

void FieldSolver::allocate_next_phi()
{
	size_t dim = mesh.n_nodes.x * mesh.n_nodes.y * mesh.n_nodes.z;
	cudaError_t cuda_status;

	cuda_status= cudaMalloc<double>(&dev_phi_next, dim);
	
}
void FieldSolver::init_constants() {

}
void FieldSolver::eval_potential(Inner_regions_manager &inner_regions)
{
	solve_poisson_eqn_Jacobi(inner_regions);
}

void FieldSolver::solve_poisson_eqn_Jacobi(Inner_regions_manager &inner_regions)
{
	max_Jacobi_iterations = 150;
	int iter;

	//init_current_phi_from_mesh_phi();
	for (iter = 0; iter < max_Jacobi_iterations; ++iter) {
		single_Jacobi_iteration(inner_regions);
		if (iterative_Jacobi_solutions_converged()) {
			break;
		}
		set_phi_next_as_phi_current();
	}
	if (iter == max_Jacobi_iterations) {
		printf("WARING: potential evaluation did't converge after max iterations!\n");
	}
	set_phi_next_as_phi_current();

	//return;
}
void FieldSolver::single_Jacobi_iteration(Inner_regions_manager &inner_regions)
{
	set_phi_next_at_boundaries();
	compute_phi_next_at_inner_points();
	set_phi_next_at_inner_regions(inner_regions);
}

void FieldSolver::set_phi_next_at_boundaries()
{
	mesh.set_boundary_conditions(dev_phi_next);
}

void FieldSolver::compute_phi_next_at_inner_points()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;

	ComputePhiNext<<<blocks, threads>>> (mesh.dev_potential, mesh.dev_charge_density, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
}

void FieldSolver::set_phi_next_at_inner_regions(Inner_regions_manager &inner_regions)
{
	//for (auto &reg : inner_regions.regions) {
	//	for (auto &node : reg.inner_nodes) {
	//		// todo: mark nodes at edge during construction
	//		// if (!node.at_domain_edge( nx, ny, nz )) {
	//		phi_next[node.x][node.y][node.z] = reg.potential;
	//		// }
	//	}
	//}
}


bool FieldSolver::iterative_Jacobi_solutions_converged()
{
	//// todo: bind tol to config parameters
	////abs_tolerance = std::max( dx * dx, std::max( dy * dy, dz * dz ) ) / 5;

	//double diff;
	//double rel_diff;
	////double tol;
	////
	//for (int i = 0; i < nx; i++) {
	//	for (int j = 0; j < ny; j++) {
	//		for (int k = 0; k < nz; k++) {
	//			diff = fabs(phi_next[i][j][k] - phi_current[i][j][k]);
	//			rel_diff = diff / fabs(phi_current[i][j][k]);
	//			if (diff > abs_tolerance || rel_diff > rel_tolerance) {
	//				return false;
	//			}
	//		}
	//	}
	//}
	//return true;
}


void FieldSolver::set_phi_next_as_phi_current()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;
	SetPhiNextAsCurrent <<<blocks, threads >>> (mesh.dev_potential, dev_phi_next);
	cuda_status = cudaDeviceSynchronize();
}


void FieldSolver::eval_fields_from_potential()
{
	dim3 threads = mesh.GetThreads();
	dim3 blocks = mesh.GetBlocks(threads);
	cudaError_t cuda_status;

	EvaluateFields <<<blocks, threads >>> (mesh.dev_potential, mesh.dev_electric_field);

	cuda_status = cudaDeviceSynchronize();
	return;
}




FieldSolver::~FieldSolver()
{
	// delete phi arrays?
	cudaFree((void*)dev_phi_next);
}