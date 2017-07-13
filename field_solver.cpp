#include "field_solver.h"

extern void solve_poisson_cuda( double *rhs,double *phi_vec, int nx, double cell_size );
extern void allocate_matrix_cuda(int nx);

Field_solver::Field_solver( Spatial_mesh &spat_mesh,
			    Inner_regions_manager &inner_regions )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    int nrows = (nx-2)*(ny-2)*(nz-2);
    int ncols = nrows;

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );    
    
    allocate_rhs(nrows);
    allocate_phi_vec(nrows);
    allocate_matrix_cuda(nx);
    
}

void Field_solver::allocate_rhs(int nrows)
{
    for (int i = 0; i<nrows; i++){
	rhs_array.push_back(0);
    }
    return;
}

void Field_solver::allocate_phi_vec(int nrows)
{
    for (int i = 0; i<nrows; i++){
	phi_array.push_back(0);
    }
    return;
}

void Field_solver::eval_potential( Spatial_mesh &spat_mesh,
				   Inner_regions_manager &inner_regions )
{
    solve_poisson_eqn( spat_mesh, inner_regions);
}

void Field_solver::solve_poisson_eqn( Spatial_mesh &spat_mesh,
				      Inner_regions_manager &inner_regions)
{

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_process_rank );
    //VecView(rhs,PETSC_VIEWER_STDOUT_WORLD);
    MPI_Barrier( MPI_COMM_WORLD );
    for( int proc = 0; proc < mpi_n_of_proc; proc++ ){

	int nx = spat_mesh.x_n_nodes;
	int ny = spat_mesh.y_n_nodes;
	int nz = spat_mesh.z_n_nodes;
	double cell_size = spat_mesh.x_cell_size;
	int i_and_j_part;
	int i,j,k;
	double *local_rhs_values;
	double *local_phi_values;
	double dx = spat_mesh.x_cell_size;
	double dy = spat_mesh.y_cell_size;
	double dz = spat_mesh.z_cell_size;
	double rhs_at_node;

	    // todo: split into separate functions
	
	for ( int k = 1; k <= nz-2; k++ ) {
	    for ( int j = 1; j <= ny-2; j++ ) { 
		for ( int i = 1; i <= nx-2; i++ ) {
		    // - 4 * pi * rho * dx^2 * dy^2
		    rhs_at_node = -4.0 * M_PI * spat_mesh.charge_density[i][j][k];
		    rhs_at_node = rhs_at_node * dx * dx * dy * dy * dz * dz;
		    // left and right boundary
		    rhs_at_node = rhs_at_node
			- dy * dy * dz * dz *
			( kronecker_delta(i,1) * spat_mesh.potential[0][j][k] +
			  kronecker_delta(i,nx-2) * spat_mesh.potential[nx-1][j][k] );
		    // top and bottom boundary
		    rhs_at_node = rhs_at_node
			- dx * dx * dz * dz *
			( kronecker_delta(j,1) * spat_mesh.potential[i][0][k] +
			  kronecker_delta(j,ny-2) * spat_mesh.potential[i][ny-1][k] );
		    // near and far boundary
		    rhs_at_node = rhs_at_node
			- dx * dx * dy * dy *
			( kronecker_delta(k,1) * spat_mesh.potential[i][j][0] +
			  kronecker_delta(k,nz-2) * spat_mesh.potential[i][j][nz-1] );
		    // set rhs vector values
		    rhs_array[node_ijk_to_global_index_in_matrix( i, j, k, nx, ny, nz )]=rhs_at_node;
		}
	    }
	}
	
	local_phi_values = &phi_array[0];
	local_rhs_values = &rhs_array[0];
	
	solve_poisson_cuda( local_rhs_values, local_phi_values, nx ,cell_size);
	for (int global_index =0; global_index<(nx-2)*(nx-2)*(nx-2); global_index++){
	        
	k = global_index / ( ( nx - 2 ) * ( ny - 2 ) ) + 1;
	i_and_j_part = global_index % ( ( nx - 2 ) * ( ny - 2 ) );
	j = i_and_j_part / ( nx - 2 ) + 1;
	i = i_and_j_part % ( nx - 2 ) + 1;
	spat_mesh.potential[i][j][k] = phi_array[global_index];
	}
    }

    
    return;
}

int Field_solver::kronecker_delta( int i,  int j )
{
    if ( i == j ) {
	return 1;
    } else {
	return 0;
    }
}

int Field_solver::node_global_index_in_matrix( Node_reference &node, int nx, int ny, int nz )
{
    return node_ijk_to_global_index_in_matrix( node.x, node.y, node.z, nx, ny, nz );
}

std::vector<int> Field_solver::list_of_nodes_global_indices_in_matrix(
    std::vector<Node_reference> &nodes,
    int nx, int ny, int nz )
{
    std::vector<int> indices;
    indices.reserve( nodes.size() );
    for( auto &node : nodes )
	indices.push_back( node_global_index_in_matrix( node, nx, ny, nz ) );
    return indices;
}

int Field_solver::node_ijk_to_global_index_in_matrix( int i, int j, int k,
						      int nx, int ny, int nz )    
{
    // numbering of nodes corresponds to axis direction
    // i.e. numbering starts from bottom-right-near corner
    //   then along X axis to the left
    //   then along Y axis to the top
    //   then along Z axis far
    if ( ( i <= 0 ) || ( i >= nx-1 ) ||
	 ( j <= 0 ) || ( j >= ny-1 ) ||
	 ( k <= 0 ) || ( k >= nz-1 ) ) {
	printf("incorrect index at node_ijk_to_global_index_in_matrix: i = %d, j=%d, k=%d \n", i,j,k);
	printf("this is not supposed to happen; aborting \n");
	exit( EXIT_FAILURE );
    } else {
	return (i - 1) + (j - 1) * ( nx - 2 ) + ( k - 1 ) * ( nx - 2 ) * ( ny - 2 );
    }    
}

void Field_solver::global_index_in_matrix_to_node_ijk( int global_index,
						       int *i, int *j, int *k,
						       int nx, int ny, int nz )
{
    // global_index = (i - 1) + (j - 1) * ( nx - 2 ) + ( k - 1 ) * ( nx - 2 ) * ( ny - 2 );
    int i_and_j_part;
    *k = global_index / ( ( nx - 2 ) * ( ny - 2 ) ) + 1;
    i_and_j_part = global_index % ( ( nx - 2 ) * ( ny - 2 ) );
    *j = i_and_j_part / ( nx - 2 ) + 1;
    *i = i_and_j_part % ( nx - 2 ) + 1;
    //todo: remove test
    // if( node_ijk_to_global_index_in_matrix( *i, *j, *k, nx, ny, nz ) != global_index ){
    // 	printf( "mistake in global_index_in_matrix_to_node_ijk; aborting" );
    // 	exit( EXIT_FAILURE );
    // }
    return;
}

void Field_solver::eval_fields_from_potential( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    boost::multi_array<double, 3> &phi = spat_mesh.potential;
    double ex, ey, ez;

    for ( int i = 0; i < nx; i++ ) {
	for ( int j = 0; j < ny; j++ ) {
	    for ( int k = 0; k < nz; k++ ) {
		if ( i == 0 ) {
		    ex = - boundary_difference( phi[i][j][k], phi[i+1][j][k], dx );
		} else if ( i == nx-1 ) {
		    ex = - boundary_difference( phi[i-1][j][k], phi[i][j][k], dx );
		} else {
		    ex = - central_difference( phi[i-1][j][k], phi[i+1][j][k], dx );
		}

		if ( j == 0 ) {
		    ey = - boundary_difference( phi[i][j][k], phi[i][j+1][k], dy );
		} else if ( j == ny-1 ) {
		    ey = - boundary_difference( phi[i][j-1][k], phi[i][j][k], dy );
		} else {
		    ey = - central_difference( phi[i][j-1][k], phi[i][j+1][k], dy );
		}

		if ( k == 0 ) {
		    ez = - boundary_difference( phi[i][j][k], phi[i][j][k+1], dz );
		} else if ( k == nz-1 ) {
		    ez = - boundary_difference( phi[i][j][k-1], phi[i][j][k], dz );
		} else {
		    ez = - central_difference( phi[i][j][k-1], phi[i][j][k+1], dz );
		}

		spat_mesh.electric_field[i][j][k] = vec3d_init( ex, ey, ez );
	    }
	}
    }

    return;
}

double Field_solver::central_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / ( 2.0 * dx ) );
}

double Field_solver::boundary_difference( double phi1, double phi2, double dx )
{    
    return ( (phi2 - phi1) / dx );
}

