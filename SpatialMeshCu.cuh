#ifndef _SPATIAL_MESH_CUH_
#define _SPATIAL_MESH_CUH_

#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "config.h"


class SpatialMeshCu {
public:

	int3 n_nodes;
	double3 cell_size;
	double3 volume_size;
	double3 *dev_node_coordinates;
	double *dev_charge_density;
	double *dev_potential;
	double3 *dev_electric_field;


	//double border_left;
	//double border_right;
	//double border_top;
	//double border_bottom;
	//double border_near;
	//double border_far;


public:
	SpatialMeshCu(Config &conf);
	SpatialMeshCu(hid_t h5_spat_mesh_group);
	void clear_old_density_values();
	void set_boundary_conditions(double* d_potential);
	bool is_potential_equal_on_boundaries();
	void print();
	void write_to_file(hid_t hdf5_file_id);
	virtual ~SpatialMeshCu();
	double node_number_to_coordinate_x(int i);
	double node_number_to_coordinate_y(int j);
	double node_number_to_coordinate_z(int k);
	dim3 GetThreads();
	dim3 GetBlocks(dim3 nThreads);
private:
	// init
	void check_correctness_of_related_config_fields(Config &conf);
	void init_constants(Config &conf);
	void copy_constants_to_device();
	void copy_boundary_to_device(Config &conf);
	void allocate_ongrid_values();
	void fill_node_coordinates();
	
	// print
	void print_grid();
	void print_ongrid_values();
	// write hdf5
	void write_hdf5_attributes(hid_t group_id);
	void write_hdf5_ongrid_values(hid_t group_id);
	int n_of_elements_to_write_for_each_process_for_1d_dataset(int total_elements);
	int data_offset_for_each_process_for_1d_dataset(int total_elements);
	void hdf5_status_check(herr_t status);
	// config check
	void grid_x_size_gt_zero(Config &conf);
	void grid_x_step_gt_zero_le_grid_x_size(Config &conf);
	void grid_y_size_gt_zero(Config &conf);
	void grid_y_step_gt_zero_le_grid_y_size(Config &conf);
	void grid_z_size_gt_zero(Config &conf);
	void grid_z_step_gt_zero_le_grid_z_size(Config &conf);
	void check_and_exit_if_not(const bool &should_be, const std::string &message);
	//cuda
	void cuda_status_check(cudaError_t status, std::string &sender);

};
#endif /* _SPATIAL_MESH_CUH_ */
