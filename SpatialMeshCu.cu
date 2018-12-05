#include "SpatialMeshCu.cuh"
#include "device_launch_parameters.h"

SpatialMeshCu::SpatialMeshCu(Config &conf)
{
	check_correctness_of_related_config_fields(conf);
	init_x_grid(conf);
	init_y_grid(conf);
	init_z_grid(conf);
	init_constants(conf);
	allocate_ongrid_values();
	fill_node_coordinates();
	set_boundary_conditions(conf);
}

void SpatialMeshCu::check_correctness_of_related_config_fields(Config &conf)
{
	grid_x_size_gt_zero(conf);
	grid_x_step_gt_zero_le_grid_x_size(conf);
	grid_y_size_gt_zero(conf);
	grid_y_step_gt_zero_le_grid_y_size(conf);
	grid_z_size_gt_zero(conf);
	grid_z_step_gt_zero_le_grid_z_size(conf);
}

void SpatialMeshCu::init_x_grid(Config &conf)
{
	//x_volume_size = conf.mesh_config_part.grid_x_size;
	//x_n_nodes =
	//	ceil(conf.mesh_config_part.grid_x_size / conf.mesh_config_part.grid_x_step) + 1;
	//x_cell_size = x_volume_size / (x_n_nodes - 1);
	//if (x_cell_size != conf.mesh_config_part.grid_x_step) {
	//	std::cout.precision(3);
	//	std::cout << "X_step was shrinked to " << x_cell_size
	//		<< " from " << conf.mesh_config_part.grid_x_step
	//		<< " to fit round number of cells" << std::endl;
	//}
	//return;
}

void SpatialMeshCu::init_y_grid(Config &conf)
{
	//y_volume_size = conf.mesh_config_part.grid_y_size;
	//y_n_nodes =
	//	ceil(conf.mesh_config_part.grid_y_size / conf.mesh_config_part.grid_y_step) + 1;
	//y_cell_size = y_volume_size / (y_n_nodes - 1);
	//if (y_cell_size != conf.mesh_config_part.grid_y_step) {
	//	std::cout.precision(3);
	//	std::cout << "Y_step was shrinked to " << y_cell_size
	//		<< " from " << conf.mesh_config_part.grid_y_step
	//		<< " to fit round number of cells." << std::endl;
	//}
	//return;
}

void SpatialMeshCu::init_z_grid(Config &conf)
{
	//z_volume_size = conf.mesh_config_part.grid_z_size;
	//z_n_nodes =
	//	ceil(conf.mesh_config_part.grid_z_size / conf.mesh_config_part.grid_z_step) + 1;
	//z_cell_size = z_volume_size / (z_n_nodes - 1);
	//if (z_cell_size != conf.mesh_config_part.grid_z_step) {
	//	std::cout.precision(3);
	//	std::cout << "Z_step was shrinked to " << z_cell_size
	//		<< " from " << conf.mesh_config_part.grid_z_step
	//		<< " to fit round number of cells." << std::endl;
	//}
	//return;
}

void SpatialMeshCu::init_constants(Config & conf)
{
	n_nodes = dim3(
		ceil(conf.mesh_config_part.grid_x_size / conf.mesh_config_part.grid_x_step) + 1,
		ceil(conf.mesh_config_part.grid_y_size / conf.mesh_config_part.grid_y_step) + 1,
		ceil(conf.mesh_config_part.grid_z_size / conf.mesh_config_part.grid_z_step) + 1
	);
	cudaMemcpyToSymbol((void*)&d_n_nodes, (void*)&n_nodes,sizeof(double3),cudaMemcpyHostToDevice);

	double3 volume_size = make_double3(
		conf.mesh_config_part.grid_x_size,
		conf.mesh_config_part.grid_y_size,
		conf.mesh_config_part.grid_z_size
	);
	cudaMemcpyToSymbol((void*)& d_volume_size, (void*)& volume_size, sizeof(double3), cudaMemcpyHostToDevice);

	double3 cell_size = make_double3(
		volume_size.x / (n_nodes.x - 1),
		volume_size.y / (n_nodes.y - 1),
		volume_size.z / (n_nodes.z - 1)
	);
	cudaMemcpyToSymbol((void*)& d_volume_size, (void*)& volume_size, sizeof(double3), cudaMemcpyHostToDevice);

	///TODO Border constants init
}

void SpatialMeshCu::allocate_ongrid_values()
{
	//TODO
	int nx = n_nodes.x;
	int ny = n_nodes.y;
	int nz = n_nodes.z;

	size_t total_node_count = nx * ny * nz;

	cudaMalloc<double3>(&dev_node_coordinates, total_node_count);
	cudaMalloc<double>(&dev_charge_density, total_node_count);
	cudaMalloc<double>(&dev_potential, total_node_count);
	cudaMalloc<double3>(&dev_electric_field, total_node_count);

	
	//node_coordinates.resize(boost::extents[nx][ny][nz]);
	//charge_density.resize(boost::extents[nx][ny][nz]);
	//potential.resize(boost::extents[nx][ny][nz]);
	//electric_field.resize(boost::extents[nx][ny][nz]);

	return;
}

void SpatialMeshCu::fill_node_coordinates()
{
	dim3 threads = GetThreads();
	dim3 blocks = GetBlocks(threads);

	<<<threads, blocks >>> fill_coordinates(dev_node_coordinates);
}

  

__global__ void SpatialMeshCu::fill_coordinates(double3* node_coordinates) {

	int idx = GetIdxVolume();

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int z = threadIdx.z + blockIdx.z*blockDim.z;
	node_coordinates[idx] = make_double3(d_volume_size.x * x, d_volume_size.y * y, d_volume_size.z * z);//(double).,
}
void SpatialMeshCu::clear_old_density_values()
{
	//std::fill(charge_density.data(),
	//	charge_density.data() + charge_density.num_elements(),
	//	0.0);

	//return;
}


void SpatialMeshCu::set_boundary_conditions(Config &conf)
{
	set_boundary_conditions(conf.boundary_config_part.boundary_phi_left,
		conf.boundary_config_part.boundary_phi_right,
		conf.boundary_config_part.boundary_phi_top,
		conf.boundary_config_part.boundary_phi_bottom,
		conf.boundary_config_part.boundary_phi_near,
		conf.boundary_config_part.boundary_phi_far);
}


void SpatialMeshCu::set_boundary_conditions(const double phi_left, const double phi_right,
	const double phi_top, const double phi_bottom,
	const double phi_near, const double phi_far)
{
	dim3 blockSize = dim3(16, 16, 2);

	dim3 gridSize = dim3(n_nodes.y / 16, n_nodes.z / 16, 1);
	<<<blockSize, gridSize >>> SetBoundaryConditionOrthoX(dev_potential);

	gridSize = dim3(n_nodes.x / 16, n_nodes.z / 16, 2);
	<<<blockSize, gridSize >>> SetBoundaryConditionOrthoY(dev_potential);

	gridSize = dim3(n_nodes.x / 16, n_nodes.y / 16, 2);
	<<<blockSize, gridSize >>> SetBoundaryConditionOrthoZ(dev_potential);
		//for (int j = 0; j < ny; j++) {
		//	for (int k = 0; k < nz; k++) {
		//		potential[0][j][k] = phi_right;
		//		potential[nx - 1][j][k] = phi_left;
		//	}
		//}

		//for (int i = 0; i < nx; i++) {
		//	for (int j = 0; j < ny; j++) {
		//		potential[i][j][0] = phi_near;
		//		potential[i][j][nz - 1] = phi_far;
		//	}
		//}

		return;
}

bool SpatialMeshCu::is_potential_equal_on_boundaries()
{
	//bool equal = (potential[0][2][2] == potential[x_n_nodes - 1][2][2] ==
	//	potential[2][0][2] == potential[2][y_n_nodes - 1][2] ==
	//	potential[2][2][0] == potential[2][2][z_n_nodes - 1]);
	// possible to rewrite to avoid warnings from compiler:
	// bool equal = ( potential[0][2][2] == potential[x_n_nodes-1][2][2] );
	// equal = equal and ( potential[x_n_nodes-1][2][2] == potential[2][0][2] );
	// equal = equal and ( potential[2][0][2] == potential[2][y_n_nodes-1][2] );
	// equal = equal and ( potential[2][y_n_nodes-1][2] == potential[2][2][0] );
	// equal = equal and ( potential[2][2][0] == potential[2][2][z_n_nodes-1] );
	//return equal;
	return false;
}

void SpatialMeshCu::print()
{
	print_grid();
	print_ongrid_values();
	return;
}

void SpatialMeshCu::print_grid()
{
	//std::cout << "Grid:" << std::endl;
	//std::cout << "Length: x = " << x_volume_size << ", "
	//	<< "y = " << y_volume_size << ", "
	//	<< "z = " << z_volume_size << std::endl;
	//std::cout << "Cell size: x = " << x_cell_size << ", "
	//	<< "y = " << y_cell_size << ", "
	//	<< "z = " << z_cell_size << std::endl;
	//std::cout << "Total nodes: x = " << x_n_nodes << ", "
	//	<< "y = " << y_n_nodes << ", "
	//	<< "z = " << z_n_nodes << std::endl;
	//return;
}

void SpatialMeshCu::print_ongrid_values()
{
	//int nx = x_n_nodes;
	//int ny = y_n_nodes;
	//int nz = z_n_nodes;
	//std::cout << "x_node, y_node, z_node, charge_density, potential, electric_field(x,y,z)" << std::endl;
	//std::cout.precision(3);
	//std::cout.setf(std::ios::scientific);
	//std::cout.fill(' ');
	//std::cout.setf(std::ios::right);
	//for (int i = 0; i < nx; i++) {
	//	for (int j = 0; j < ny; j++) {
	//		for (int k = 0; k < nz; k++) {
	//			std::cout << std::setw(8) << i
	//				<< std::setw(8) << j
	//				<< std::setw(8) << k
	//				<< std::setw(14) << charge_density[i][j][k]
	//				<< std::setw(14) << potential[i][j][k]
	//				<< std::setw(14) << vec3d_x(electric_field[i][j][k])
	//				<< std::setw(14) << vec3d_y(electric_field[i][j][k])
	//				<< std::setw(14) << vec3d_z(electric_field[i][j][k])
	//				<< std::endl;
	//		}
	//	}
	//}
	//return;
}

void SpatialMeshCu::grid_x_size_gt_zero(Config &conf)
{
	check_and_exit_if_not(conf.mesh_config_part.grid_x_size > 0,
		"grid_x_size < 0");
}

void SpatialMeshCu::grid_x_step_gt_zero_le_grid_x_size(Config &conf)
{
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_x_step > 0) &&
		(conf.mesh_config_part.grid_x_step <= conf.mesh_config_part.grid_x_size),
		"grid_x_step < 0 or grid_x_step >= grid_x_size");
}

void SpatialMeshCu::grid_y_size_gt_zero(Config &conf)
{
	check_and_exit_if_not(conf.mesh_config_part.grid_y_size > 0,
		"grid_y_size < 0");
}

void SpatialMeshCu::grid_y_step_gt_zero_le_grid_y_size(Config &conf)
{
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_y_step > 0) &&
		(conf.mesh_config_part.grid_y_step <= conf.mesh_config_part.grid_y_size),
		"grid_y_step < 0 or grid_y_step >= grid_y_size");
}

void SpatialMeshCu::grid_z_size_gt_zero(Config &conf)
{
	check_and_exit_if_not(conf.mesh_config_part.grid_z_size > 0,
		"grid_z_size < 0");
}

void SpatialMeshCu::grid_z_step_gt_zero_le_grid_z_size(Config &conf)
{
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_z_step > 0) &&
		(conf.mesh_config_part.grid_z_step <= conf.mesh_config_part.grid_z_size),
		"grid_z_step < 0 or grid_z_step >= grid_z_size");
}


void SpatialMeshCu::check_and_exit_if_not(const bool &should_be, const std::string &message)
{
	//if (!should_be) {
	//	std::cout << "Error: " << message << std::endl;
	//	exit(EXIT_FAILURE);
	//}
	//return;
}

double SpatialMeshCu::node_number_to_coordinate_x(int i)
{
	//if (i >= 0 && i < x_n_nodes) {
	//	return i * x_cell_size;
	//}
	//else {
	//	printf("invalid node number i=%d at node_number_to_coordinate_x\n", i);
	//	exit(EXIT_FAILURE);
	//}
}

double SpatialMeshCu::node_number_to_coordinate_y(int j)
{
	//if (j >= 0 && j < y_n_nodes) {
	//	return j * y_cell_size;
	//}
	//else {
	//	printf("invalid node number j=%d at node_number_to_coordinate_y\n", j);
	//	exit(EXIT_FAILURE);
	//}
}

double SpatialMeshCu::node_number_to_coordinate_z(int k)
{
	//if (k >= 0 && k < z_n_nodes) {
	//	return k * z_cell_size;
	//}
	//else {
	//	printf("invalid node number k=%d at node_number_to_coordinate_z\n", k);
	//	exit(EXIT_FAILURE);
	//}
}

dim3 SpatialMeshCu::GetThreads() {
	return dim3(16, 16, d_n_nodes.z / 16);
}

dim3 SpatialMeshCu::GetBlocks(dim3 nThreads) {
	return dim3(d_n_nodes.x / nThreads.x, d_n_nodes.y / nThreads.y, 16);
}

__device__ int SpatialMeshCu::GetIdxVolume() {
	//int xStepthread = 1;
	int xStepBlock = blockDim.x;

	int yStepThread = d_n_nodes.x;
	int yStepBlock = yStepThread * blockDim.y;

	int zStepThread = d_n_nodes.x * d_n_nodes.y;
	int zStepBlock = zStepThread * blockDim.z;

	return threadIdx.x + blockIdx.x*xStepBlock +
		threadIdx.y*yStepThread + blockIdx.y*yStepBlock +
		threadIdx.z*zStepThread + blockIdx.z*zStepBlock;
}

__global__ void SpatialMeshCu::SetBoundaryConditionOrthoX(double* potential) {
	int xIdx = blockIdx.z*(d_n_nodes.x - 1); //0 or nodes.x-1


	int yStepThread = d_n_nodes.x;//x=
	int yStepBlock = d_n_nodes.x * blockDim.x;

	int zStepThread = d_n_nodes.x * d_n_nodes.y;
	int zStepBlock = zStepThread * blockDim.y;

	int idx = xIdx +
		threadIdx.x*yStepThread + blockIdx.x*yStepBlock +
		threadIdx.y*zStepThread + blockIdx.y*zStepBlock;

	potential[idx] = ((double)(1 - blockIdx.z))*d_bot_border + (blockIdx.z*d_up_border);

}

//верхн€€ и нижн€€ граница устанавливаютс€ одним вызовом
__global__ void SpatialMeshCu::SetBoundaryConditionOrthoX(double* potential) {
	int xIdx = blockIdx.z*(d_n_nodes.x - 1); //0 or nodes.x-1


	int yStepThread = d_n_nodes.x;//x=
	int yStepBlock = d_n_nodes.x * blockDim.x;

	int zStepThread = d_n_nodes.x * d_n_nodes.y;
	int zStepBlock = zStepThread * blockDim.y;

	int idx = xIdx +
		threadIdx.x*yStepThread + blockIdx.x*yStepBlock +
		threadIdx.y*zStepThread + blockIdx.y*zStepBlock;
	//используетс€ дл€ замены if конструкции
	potential[idx] = ((double)(1 - blockIdx.z)) * d_left_border + (blockIdx.z*d_right_border);

}

__global__ void SpatialMeshCu::SetBoundaryConditionOrthoY(double* potential) {
	int yIdx = blockIdx.z * d_n_nodes.x*(d_n_nodes.y - 1); //0 or nodes.x-1


	int xStepThread = 1;//x=
	int xStepBlock = blockDim.x;

	int zStepThread = d_n_nodes.x * d_n_nodes.y;
	int zStepBlock = zStepThread * blockDim.y;

	int idx = yIdx +
		threadIdx.x*xStepThread + blockIdx.x*xStepBlock +
		threadIdx.y*zStepThread + blockIdx.y*zStepBlock;
	//используетс€ дл€ замены if конструкции
	potential[idx] = ((double)(1 - blockIdx.z)) * d_bot_border + (blockIdx.z * d_up_border);

}

__global__ void SpatialMeshCu::SetBoundaryConditionOrthoZ(double* potential) {
	int zIdx = blockIdx.z * (d_n_nodes.x * d_n_nodes.y * (d_n_nodes.z - 1)); //0 or nodes.x-1


	int xStepThread = 1;//x=
	int xStepBlock = blockDim.x;

	int yStepThread = d_n_nodes.x;
	int yStepBlock = yStepThread * blockDim.y;

	int idx = zIdx +
		threadIdx.x*xStepThread + blockIdx.x*xStepBlock +
		threadIdx.y*yStepThread + blockIdx.y*yStepBlock;
	//используетс€ дл€ замены if конструкции
	potential[idx] = ((double)(1 - blockIdx.z)) * d_near_border + (blockIdx.z * d_far_border);

}

SpatialMeshCu::~SpatialMeshCu() {
	cudaFree((void*)dev_node_coordinates);
	cudaFree((void*)dev_potential);
	cudaFree((void*)dev_charge_density);
	cudaFree((void*)dev_electric_field);
}
