#include "SpatialMeshCu.cuh"

__constant__ double3 d_volume_size[1];
__constant__ double3 d_cell_size[1];
__constant__ int3 d_n_nodes[1];
__constant__ double d_boundary[6];

#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define FAR 4
#define NEAR 5

__device__ int GetIdxVolume() {
	//int xStepthread = 1;
	int xStepBlock = blockDim.x;

	int yStepThread = d_n_nodes[0].x;
	int yStepBlock = yStepThread * blockDim.y;

	int zStepThread = d_n_nodes[0].x * d_n_nodes[0].y;
	int zStepBlock = zStepThread * blockDim.z;

	return threadIdx.x + blockIdx.x * xStepBlock + threadIdx.y * yStepThread
		+ blockIdx.y * yStepBlock + threadIdx.z * zStepThread
		+ blockIdx.z * zStepBlock;
}

__global__ void fill_coordinates(double3* node_coordinates) {

	int idx = GetIdxVolume();

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	node_coordinates[idx] = make_double3(d_volume_size[0].x * x,
		d_volume_size[0].y * y, d_volume_size[0].z * z); //(double).,
}

__global__ void SetBoundaryConditionOrthoX(double* potential) {
	int xIdx = blockIdx.z * (d_n_nodes[0].x - 1); //0 or nodes.x-1

	int yStepThread = d_n_nodes[0].x; //x=
	int yStepBlock = d_n_nodes[0].x * blockDim.x;

	int zStepThread = d_n_nodes[0].x * d_n_nodes[0].y;
	int zStepBlock = zStepThread * blockDim.y;

	int idx = xIdx + threadIdx.x * yStepThread + blockIdx.x * yStepBlock
		+ threadIdx.y * zStepThread + blockIdx.y * zStepBlock;

	potential[idx] = ((double)(1 - blockIdx.z)) * d_boundary[LEFT]
		+ (blockIdx.z * d_boundary[RIGHT]);

}

__global__ void SetBoundaryConditionOrthoY(double* potential) {
	int yIdx = blockIdx.z * d_n_nodes[0].x * (d_n_nodes[0].y - 1); //0 or nodes.x-1

	int xStepThread = 1; //x=
	int xStepBlock = blockDim.x;

	int zStepThread = d_n_nodes[0].x * d_n_nodes[0].y;
	int zStepBlock = zStepThread * blockDim.y;

	int idx = yIdx + threadIdx.x * xStepThread + blockIdx.x * xStepBlock
		+ threadIdx.y * zStepThread + blockIdx.y * zStepBlock;

	potential[idx] = ((double)(1 - blockIdx.z)) * d_boundary[BOTTOM]
		+ (blockIdx.z * d_boundary[TOP]);

}

__global__ void SetBoundaryConditionOrthoZ(double* potential) {
	int zIdx = blockIdx.z
		* (d_n_nodes[0].x * d_n_nodes[0].y * (d_n_nodes[0].z - 1)); //0 or nodes.x-1

	int xStepThread = 1; //x=
	int xStepBlock = blockDim.x;

	int yStepThread = d_n_nodes[0].x;
	int yStepBlock = yStepThread * blockDim.y;

	int idx = zIdx + threadIdx.x * xStepThread + blockIdx.x * xStepBlock
		+ threadIdx.y * yStepThread + blockIdx.y * yStepBlock;
	potential[idx] = ((double)(1 - blockIdx.z)) * d_boundary[NEAR]
		+ (blockIdx.z * d_boundary[FAR]);

}

SpatialMeshCu::SpatialMeshCu(Config &conf) {
	check_correctness_of_related_config_fields(conf);
	init_constants(conf);
	allocate_ongrid_values();
	fill_node_coordinates();
	set_boundary_conditions(dev_potential);
}

SpatialMeshCu::SpatialMeshCu(hid_t h5_spat_mesh_group) {
	herr_t status;
	cudaError_t cuda_status;
	std::string debug_message = std::string(" reading from hdf5 ");

	volume_size = make_double3(0, 0, 0);
	cell_size = make_double3(0, 0, 0);
	n_nodes = make_int3(0, 0, 0);

	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "x_volume_size",
		&volume_size.x);
	hdf5_status_check(status);
	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "y_volume_size",
		&volume_size.y);
	hdf5_status_check(status);
	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "z_volume_size",
		&volume_size.z);
	hdf5_status_check(status);

	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "x_cell_size",
		&cell_size.x);
	hdf5_status_check(status);
	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "y_cell_size",
		&cell_size.y);
	hdf5_status_check(status);
	status = H5LTget_attribute_double(h5_spat_mesh_group, "./", "z_cell_size",
		&cell_size.z);
	hdf5_status_check(status);

	status = H5LTget_attribute_int(h5_spat_mesh_group, "./", "x_n_nodes",
		&n_nodes.x);
	hdf5_status_check(status);
	status = H5LTget_attribute_int(h5_spat_mesh_group, "./", "y_n_nodes",
		&n_nodes.y);
	hdf5_status_check(status);
	status = H5LTget_attribute_int(h5_spat_mesh_group, "./", "z_n_nodes",
		&n_nodes.z);
	hdf5_status_check(status);

	allocate_ongrid_values();
	copy_constants_to_device();

	int dim = n_nodes.x * n_nodes.y * n_nodes.z;
	double *h5_tmp_buf_1 = new double[dim];
	double *h5_tmp_buf_2 = new double[dim];
	double *h5_tmp_buf_3 = new double[dim];

	dim3 threads = GetThreads();
	dim3 blocks = GetBlocks(threads);
	fill_coordinates <<< blocks, threads >>> (dev_node_coordinates);
	cuda_status = cudaDeviceSynchronize();
	cuda_status_check(cuda_status, debug_message);

	H5LTread_dataset_double(h5_spat_mesh_group, "./charge_density",
		h5_tmp_buf_1);
	H5LTread_dataset_double(h5_spat_mesh_group, "./potential", h5_tmp_buf_2);

	//    for ( int i = 0; i < dim; i++ ) {
	//	( charge_density.data() )[i] = h5_tmp_buf_1[i];
	//	( potential.data() )[i] = h5_tmp_buf_2[i];
	//    }

	cuda_status = cudaMemcpy(h5_tmp_buf_1, dev_charge_density, sizeof(double) * dim,
		cudaMemcpyHostToDevice);
	cuda_status_check(cuda_status, debug_message);

	cuda_status = cudaMemcpy(h5_tmp_buf_2, dev_potential, sizeof(double) * dim,
		cudaMemcpyHostToDevice);
	cuda_status_check(cuda_status, debug_message);

	double3 *h5_tmp_vector = new double3[dim];

	H5LTread_dataset_double(h5_spat_mesh_group, "./electric_field_x",
		h5_tmp_buf_1);
	H5LTread_dataset_double(h5_spat_mesh_group, "./electric_field_y",
		h5_tmp_buf_2);
	H5LTread_dataset_double(h5_spat_mesh_group, "./electric_field_z",
		h5_tmp_buf_3);
	for (int i = 0; i < dim; i++) {
		h5_tmp_vector[i] = make_double3(h5_tmp_buf_1[i], h5_tmp_buf_2[i],
			h5_tmp_buf_3[i]);
	}

	cuda_status = cudaMemcpy(h5_tmp_buf_2, dev_electric_field, sizeof(double3) * dim,
		cudaMemcpyHostToDevice);
	cuda_status_check(cuda_status, debug_message);

	delete[] h5_tmp_buf_1;
	delete[] h5_tmp_buf_2;
	delete[] h5_tmp_buf_3;
	delete[] h5_tmp_vector;

	return;
}

void SpatialMeshCu::check_correctness_of_related_config_fields(Config &conf) {
	grid_x_size_gt_zero(conf);
	grid_x_step_gt_zero_le_grid_x_size(conf);
	grid_y_size_gt_zero(conf);
	grid_y_step_gt_zero_le_grid_y_size(conf);
	grid_z_size_gt_zero(conf);
	grid_z_step_gt_zero_le_grid_z_size(conf);
}

void SpatialMeshCu::init_constants(Config & conf) {
	n_nodes = make_int3(
		ceil(
			conf.mesh_config_part.grid_x_size
			/ conf.mesh_config_part.grid_x_step) + 1,
		ceil(
			conf.mesh_config_part.grid_y_size
			/ conf.mesh_config_part.grid_y_step) + 1,
		ceil(
			conf.mesh_config_part.grid_z_size
			/ conf.mesh_config_part.grid_z_step) + 1);

	volume_size = make_double3(conf.mesh_config_part.grid_x_size,
		conf.mesh_config_part.grid_y_size,
		conf.mesh_config_part.grid_z_size);

	cell_size = make_double3(volume_size.x / (n_nodes.x - 1),
		volume_size.y / (n_nodes.y - 1), volume_size.z / (n_nodes.z - 1));


	copy_constants_to_device();
	copy_boundary_to_device(conf);
}

void SpatialMeshCu::copy_constants_to_device() {
	cudaError_t cuda_status;
	//mesh params
	const int3 *nodes = &n_nodes;
	std::string debug_message = std::string(" copy nodes number ");
	cuda_status = cudaMemcpyToSymbol(d_n_nodes, (const void*)nodes, sizeof(int3));
	cuda_status_check(cuda_status, debug_message);

	const double3 *volume = &volume_size;
	debug_message = std::string(" copy volume size ");
	cuda_status = cudaMemcpyToSymbol(d_volume_size, (const void*)&volume, sizeof(double3));
	cuda_status_check(cuda_status, debug_message);
	
	debug_message = std::string(" copy cell size ");
	cuda_status = cudaMemcpyToSymbol(d_cell_size, (const void*)&cell_size, sizeof(double3));
	cuda_status_check(cuda_status, debug_message);

	return;
}

void SpatialMeshCu::copy_boundary_to_device(Config &conf) {
	cudaError_t cuda_status;
	//boundary params
	std::string debug_message = std::string(" copy border constants ");
	double boundary[6];
	boundary[RIGHT] = conf.boundary_config_part.boundary_phi_right;
	boundary[LEFT] = conf.boundary_config_part.boundary_phi_left;
	boundary[TOP] = conf.boundary_config_part.boundary_phi_top;
	boundary[BOTTOM] = conf.boundary_config_part.boundary_phi_bottom;
	boundary[NEAR] = conf.boundary_config_part.boundary_phi_near;
	boundary[FAR] = conf.boundary_config_part.boundary_phi_far;
	const double *c_boundary = boundary;
	cuda_status = cudaMemcpyToSymbol(d_boundary, (const void*)c_boundary,
		sizeof(double)*6);
	cuda_status_check(cuda_status, debug_message);
}

void SpatialMeshCu::allocate_ongrid_values() {
	int nx = n_nodes.x;
	int ny = n_nodes.y;
	int nz = n_nodes.z;

	size_t total_node_count = nx * ny * nz;
	cudaError_t cuda_status;

	std::string debug_message = std::string(" malloc coords");

	cuda_status = cudaMalloc < double3 >(&dev_node_coordinates, sizeof(double3) * total_node_count);
	cuda_status_check(cuda_status, debug_message);

	debug_message = std::string(" malloc charde density");
	cuda_status = cudaMalloc<double>(&dev_charge_density, sizeof(double) * total_node_count);
	cuda_status_check(cuda_status, debug_message);

	debug_message = std::string(" malloc potential");
	cuda_status = cudaMalloc<double>(&dev_potential, sizeof(double) * total_node_count);
	cuda_status_check(cuda_status, debug_message);

	debug_message = std::string(" malloc field");
	cuda_status = cudaMalloc < double3 >(&dev_electric_field, sizeof(double3) * total_node_count);
	cuda_status_check(cuda_status, debug_message);

	return;
}

void SpatialMeshCu::fill_node_coordinates() {
	dim3 threads = GetThreads();
	dim3 blocks = GetBlocks(threads);
	cudaError_t cuda_status;
	std::string debug_message = std::string(" fill coordinates ");
	fill_coordinates <<< blocks,threads>>> (dev_node_coordinates);
	cuda_status= cudaDeviceSynchronize();
	cuda_status_check(cuda_status, debug_message);

	return;
}

void SpatialMeshCu::clear_old_density_values() {
	//std::fill(charge_density.data(),
	//	charge_density.data() + charge_density.num_elements(),
	//	0.0);

	//return;
}

void SpatialMeshCu::set_boundary_conditions(double* d_potential) {
	dim3 threads = dim3(4, 4, 2);
	cudaError_t cuda_status;
	std::string debug_message = std::string(" set boundary ");

	dim3 blocks = dim3(n_nodes.y / 4, n_nodes.z / 4, 1);
	SetBoundaryConditionOrthoX <<< blocks, threads >>> (d_potential);
	cuda_status = cudaDeviceSynchronize();
	cuda_status_check(cuda_status, debug_message);

	blocks = dim3(n_nodes.x / 4, n_nodes.z / 4, 2);
	SetBoundaryConditionOrthoY <<< blocks, threads >>> (d_potential);
	cuda_status = cudaDeviceSynchronize();
	cuda_status_check(cuda_status, debug_message);

	blocks = dim3(n_nodes.x / 4, n_nodes.y / 4, 2);
	SetBoundaryConditionOrthoZ <<< blocks, threads >>> (d_potential);
	cuda_status = cudaDeviceSynchronize();
	cuda_status_check(cuda_status, debug_message);

	return;
}

bool SpatialMeshCu::is_potential_equal_on_boundaries() {
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

void SpatialMeshCu::print() {
	print_grid();
	print_ongrid_values();
	return;
}

void SpatialMeshCu::print_grid() {
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

void SpatialMeshCu::print_ongrid_values() {
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

void SpatialMeshCu::write_to_file(hid_t hdf5_file_id) {
	hid_t group_id;
	herr_t status;
	std::string hdf5_groupname = "/SpatialMesh";
	group_id = H5Gcreate(hdf5_file_id, hdf5_groupname.c_str(), H5P_DEFAULT,
		H5P_DEFAULT, H5P_DEFAULT);
	hdf5_status_check(group_id);

	write_hdf5_attributes(group_id);
	write_hdf5_ongrid_values(group_id);

	status = H5Gclose(group_id);
	hdf5_status_check(status);
	return;
}

void SpatialMeshCu::write_hdf5_attributes(hid_t group_id) {
	herr_t status;
	int single_element = 1;
	std::string hdf5_current_group = "./";

	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"x_volume_size", &volume_size.x, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"y_volume_size", &volume_size.y, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"z_volume_size", &volume_size.z, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"x_cell_size", &cell_size.x, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"y_cell_size", &cell_size.y, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_double(group_id, hdf5_current_group.c_str(),
		"z_cell_size", &cell_size.z, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_int(group_id, hdf5_current_group.c_str(),
		"x_n_nodes", &n_nodes.x, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_int(group_id, hdf5_current_group.c_str(),
		"y_n_nodes", &n_nodes.y, single_element);
	hdf5_status_check(status);
	status = H5LTset_attribute_int(group_id, hdf5_current_group.c_str(),
		"z_n_nodes", &n_nodes.z, single_element);
	hdf5_status_check(status);
}

void SpatialMeshCu::write_hdf5_ongrid_values(hid_t group_id) {
	hid_t filespace, dset;
	herr_t status;
	cudaError_t cuda_status;
	std::string debug_message = std::string(" write hdf5 ");

	int rank = 1;
	hsize_t dims[rank];
	dims[0] = n_nodes.x * n_nodes.y * n_nodes.z;

	filespace = H5Screate_simple(rank, dims, NULL);

	// todo: without compound datasets
	// there is this copying problem.
	{
		double *nx = new double[dims[0]];
		double *ny = new double[dims[0]];
		double *nz = new double[dims[0]];

		debug_message = std::string(" write hdf5 node_coords");

		double3 *hdf5_tmp_write_data = new double3[dims[0]];
		cuda_status = cudaMemcpy(hdf5_tmp_write_data, dev_node_coordinates,
			sizeof(double3) * dims[0], cudaMemcpyDeviceToHost);
		cuda_status_check(cuda_status, debug_message);
		for (unsigned int i = 0; i < dims[0]; i++) {
			nx[i] = hdf5_tmp_write_data[i].x;
			ny[i] = hdf5_tmp_write_data[i].y;
			nz[i] = hdf5_tmp_write_data[i].z;
		}

		dset = H5Dcreate(group_id, "./node_coordinates_x", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, nx);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);

		dset = H5Dcreate(group_id, "./node_coordinates_y", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, ny);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);

		dset = H5Dcreate(group_id, "./node_coordinates_z", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, nz);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);
		delete[] nx;
		delete[] ny;
		delete[] nz;
		delete[] hdf5_tmp_write_data;
	}
	{
		debug_message = std::string(" write hdf5 charge_density ");

		double *hdf5_tmp_write_data = new double[dims[0]];
		cuda_status = cudaMemcpy(hdf5_tmp_write_data, dev_charge_density,
			sizeof(double) * dims[0], cudaMemcpyDeviceToHost);
		cuda_status_check(cuda_status, debug_message);

		dset = H5Dcreate(group_id, "./charge_density", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, hdf5_tmp_write_data);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);
		delete[] hdf5_tmp_write_data;
	}
	{

		debug_message = std::string(" write hdf5 potential ");
		double *hdf5_tmp_write_data = new double[dims[0]];
		cuda_status = cudaMemcpy(hdf5_tmp_write_data, dev_potential, sizeof(double) * dims[0],
			cudaMemcpyDeviceToHost);
		cuda_status_check(cuda_status, debug_message);

		dset = H5Dcreate(group_id, "./potential", H5T_IEEE_F64BE, filespace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, hdf5_tmp_write_data);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);
		delete[] hdf5_tmp_write_data;

	}
	{
		debug_message = std::string(" write electric field to hdf5 ");

		double *ex = new double[dims[0]];
		double *ey = new double[dims[0]];
		double *ez = new double[dims[0]];
		double3 *hdf5_tmp_write_data = new double3[dims[0]];
		cuda_status = cudaMemcpy(hdf5_tmp_write_data, dev_electric_field,
			sizeof(double3) * dims[0], cudaMemcpyDeviceToHost);
		cuda_status_check(cuda_status, debug_message);

		for (unsigned int i = 0; i < dims[0]; i++) {
			ex[i] = hdf5_tmp_write_data[i].x;
			ey[i] = hdf5_tmp_write_data[i].y;
			ez[i] = hdf5_tmp_write_data[i].z;
		}
		dset = H5Dcreate(group_id, "./electric_field_x", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, ex);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);

		dset = H5Dcreate(group_id, "./electric_field_y", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, ey);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);

		dset = H5Dcreate(group_id, "./electric_field_z", H5T_IEEE_F64BE,
			filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		hdf5_status_check(dset);
		status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, filespace,
			H5P_DEFAULT, ez);
		hdf5_status_check(status);
		status = H5Dclose(dset);
		hdf5_status_check(status);
		delete[] ex;
		delete[] ey;
		delete[] ez;
		delete[] hdf5_tmp_write_data;
	}
	status = H5Sclose(filespace);
	hdf5_status_check(status);
}

void SpatialMeshCu::grid_x_size_gt_zero(Config &conf) {
	check_and_exit_if_not(conf.mesh_config_part.grid_x_size > 0,
		"grid_x_size < 0");
}

void SpatialMeshCu::grid_x_step_gt_zero_le_grid_x_size(Config &conf) {
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_x_step > 0)
		&& (conf.mesh_config_part.grid_x_step
			<= conf.mesh_config_part.grid_x_size),
		"grid_x_step < 0 or grid_x_step >= grid_x_size");
}

void SpatialMeshCu::grid_y_size_gt_zero(Config &conf) {
	check_and_exit_if_not(conf.mesh_config_part.grid_y_size > 0,
		"grid_y_size < 0");
}

void SpatialMeshCu::grid_y_step_gt_zero_le_grid_y_size(Config &conf) {
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_y_step > 0)
		&& (conf.mesh_config_part.grid_y_step
			<= conf.mesh_config_part.grid_y_size),
		"grid_y_step < 0 or grid_y_step >= grid_y_size");
}

void SpatialMeshCu::grid_z_size_gt_zero(Config &conf) {
	check_and_exit_if_not(conf.mesh_config_part.grid_z_size > 0,
		"grid_z_size < 0");
}

void SpatialMeshCu::grid_z_step_gt_zero_le_grid_z_size(Config &conf) {
	check_and_exit_if_not(
		(conf.mesh_config_part.grid_z_step > 0)
		&& (conf.mesh_config_part.grid_z_step
			<= conf.mesh_config_part.grid_z_size),
		"grid_z_step < 0 or grid_z_step >= grid_z_size");
}

void SpatialMeshCu::check_and_exit_if_not(const bool &should_be,
	const std::string &message) {
	//if (!should_be) {
	//	std::cout << "Error: " << message << std::endl;
	//	exit(EXIT_FAILURE);
	//}
	//return;
}

double SpatialMeshCu::node_number_to_coordinate_x(int i) {
	if (i >= 0 && i < n_nodes.x) {
		return i * cell_size.x;
	}
	else {
		printf("invalid node number i=%d at node_number_to_coordinate_x\n", i);
		exit(EXIT_FAILURE);
	}
	return 0;
}

double SpatialMeshCu::node_number_to_coordinate_y(int j) {
	if (j >= 0 && j < n_nodes.y) {
		return j * cell_size.y;
	}
	else {
		printf("invalid node number j=%d at node_number_to_coordinate_y\n", j);
		exit(EXIT_FAILURE);
	}
	return 0;
}

double SpatialMeshCu::node_number_to_coordinate_z(int k) {
	if (k >= 0 && k < n_nodes.z) {
		return k * cell_size.z;
	}
	else {
		printf("invalid node number k=%d at node_number_to_coordinate_z\n", k);
		exit(EXIT_FAILURE);
	}
	return 0;
}

void SpatialMeshCu::hdf5_status_check(herr_t status)
{
	if (status < 0) {
		std::cout << "Something went wrong while writing Spatial_mesh group. Aborting."
			<< std::endl;
		exit(EXIT_FAILURE);
	}
}

void SpatialMeshCu::cuda_status_check(cudaError_t status, std::string &sender)
{
	if (status > 0) {
		std::cout << "Cuda error at"<< sender <<": " << cudaGetErrorString(status) << std::endl;
		exit(EXIT_FAILURE);
	}
}

dim3 SpatialMeshCu::GetThreads() {
	return dim3(4, 4, 4);
}

dim3 SpatialMeshCu::GetBlocks(dim3 nThreads) {	
	return dim3(n_nodes.x / nThreads.x, n_nodes.y / nThreads.y, n_nodes.z/nThreads.z);
}

SpatialMeshCu::~SpatialMeshCu() {
	cudaFree((void*)dev_node_coordinates);
	cudaFree((void*)dev_potential);
	cudaFree((void*)dev_charge_density);
	cudaFree((void*)dev_electric_field);
}
