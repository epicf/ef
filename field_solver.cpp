#include "field_solver.h"

Field_solver::Field_solver( Spatial_mesh &spat_mesh,
			    Inner_regions_manager &inner_regions )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    PetscInt nrows = (nx-2)*(ny-2)*(nz-2);
    PetscInt ncols = nrows;

    PetscErrorCode ierr;
    PetscInt A_approx_nonzero_per_row = 7;

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( PETSC_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_process_rank );    
    
    alloc_petsc_vector( &phi_vec, nrows, "Solution" );
    ierr = VecSet( phi_vec, 0.0 ); CHKERRXX( ierr );
    get_vector_ownership_range_and_local_size_for_each_process( &phi_vec, &rstart, &rend, &nlocal );
    alloc_petsc_vector( &rhs, nrows, "RHS" );

    alloc_petsc_matrix( &A, nlocal, nlocal, nrows, ncols, A_approx_nonzero_per_row );
    
    construct_equation_matrix( &A, spat_mesh, inner_regions, nlocal, rstart, rend );
    create_solver_and_preconditioner( &ksp, &pc, &A );
}

void Field_solver::alloc_petsc_vector( Vec *x, int size, const char *name )
{
    PetscErrorCode ierr;
    ierr = VecCreate( PETSC_COMM_WORLD, x ); CHKERRXX( ierr );
    ierr = PetscObjectSetName( (PetscObject) *x, name ); CHKERRXX( ierr );
    ierr = VecSetSizes( *x, PETSC_DECIDE, size ); CHKERRXX( ierr );
    ierr = VecSetFromOptions( *x ); CHKERRXX( ierr );
    return;
}

void Field_solver::get_vector_ownership_range_and_local_size_for_each_process(
    Vec *x, PetscInt *rstart, PetscInt *rend, PetscInt *nlocal )
{
    PetscErrorCode ierr;
    ierr = VecGetOwnershipRange( *x, rstart, rend ); CHKERRXX(ierr);
    ierr = VecGetLocalSize( *x, nlocal ); CHKERRXX(ierr);
    return;
}

void Field_solver::alloc_petsc_matrix( Mat *A,
				       PetscInt nrow_local, PetscInt ncol_local,
				       PetscInt nrow, PetscInt ncol,
				       PetscInt nonzero_per_row )
{
    PetscErrorCode ierr;
    // PetscInt approx_nonzero_per_row = 7;

    ierr = MatCreate( PETSC_COMM_WORLD, A ); CHKERRXX( ierr );
    ierr = MatSetSizes( *A, nrow_local, ncol_local, nrow, ncol ); CHKERRXX( ierr );
    ierr = MatSetFromOptions( *A ); CHKERRXX( ierr );
    ierr = MatSetType( *A, MATAIJ ); CHKERRXX( ierr );
    // redo; set nonzero_per_row more accurately
    // if           nlocal >= (nx-2)*(ny-2): max_diag_nonzero_per_row = 7, max_offdiag_nonzer_per_row = 3
    //    (nx-2) <= nlocal < (nx-2)*(ny-2) : max_diag_nonzero_per_row = 5, max_offdiag_nonzer_per_row = 4
    // probably. 
    ierr = MatMPIAIJSetPreallocation( *A, nonzero_per_row, NULL, nonzero_per_row, NULL); CHKERRXX( ierr ); 
    ierr = MatSetUp( *A ); CHKERRXX( ierr );
    return;
}

void Field_solver::alloc_petsc_matrix_seqaij( Mat *A, PetscInt nrow,
					      PetscInt ncol, PetscInt nonzero_per_row )
{
    PetscErrorCode ierr;
    ierr = MatCreateSeqAIJ( PETSC_COMM_SELF, nrow, ncol,
    			    nonzero_per_row, NULL,  A ); CHKERRXX( ierr );
    ierr = MatSetUp( *A ); CHKERRXX( ierr );
    return;
}


void Field_solver::construct_equation_matrix( Mat *A,
					      Spatial_mesh &spat_mesh,
					      Inner_regions_manager &inner_regions,
					      PetscInt nlocal, PetscInt rstart, PetscInt rend )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;    
    
    construct_equation_matrix_in_full_domain( A, nx, ny, nz, dx, dy, dz, nlocal, rstart, rend );
    cross_out_nodes_occupied_by_objects( A, nx, ny, nz, inner_regions );
    modify_equation_near_object_boundaries( A, nx, ny, nz, dx, dy, dz, inner_regions );
}


void Field_solver::construct_equation_matrix_in_full_domain( Mat *A,
							     int nx, int ny, int nz,
							     double dx, double dy, double dz,
							     PetscInt nlocal, PetscInt rstart, PetscInt rend )
{
    PetscErrorCode ierr;
    Mat d2dy2, d2dz2;
    int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    int ncol = nrow;
    PetscInt nonzero_per_row = 7; // approx

    construct_d2dx2_in_3d( A, nx, ny, nz, rstart, rend );
    ierr = MatScale( *A, dy * dy * dz * dz ); CHKERRXX( ierr );
    
    alloc_petsc_matrix( &d2dy2, nlocal, nlocal, nrow, ncol, nonzero_per_row );
    construct_d2dy2_in_3d( &d2dy2, nx, ny, nz, rstart, rend );
    ierr = MatAXPY( *A, dx * dx * dz * dz, d2dy2, DIFFERENT_NONZERO_PATTERN ); CHKERRXX( ierr );
    ierr = MatDestroy( &d2dy2 ); CHKERRXX( ierr );

    alloc_petsc_matrix( &d2dz2, nlocal, nlocal, nrow, ncol, nonzero_per_row );
    construct_d2dz2_in_3d( &d2dz2, nx, ny, nz, rstart, rend );
    ierr = MatAXPY( *A, dx * dx * dy * dy, d2dz2, DIFFERENT_NONZERO_PATTERN ); CHKERRXX( ierr );
    ierr = MatDestroy( &d2dz2 ); CHKERRXX( ierr );

    return;
}

void Field_solver::cross_out_nodes_occupied_by_objects( Mat *A,
							int nx, int ny, int nz,
							Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions )
	cross_out_nodes_occupied_by_objects( A, nx, ny, nz, reg );
}


void Field_solver::cross_out_nodes_occupied_by_objects( Mat *A,
							int nx, int ny, int nz,
							Inner_region &inner_region )
{
    std::vector<int> occupied_nodes_global_indices =
	list_of_nodes_global_indices_in_matrix( inner_region.inner_nodes_not_at_domain_edge, nx, ny, nz );
    
    PetscErrorCode ierr;
    PetscInt num_of_rows_to_remove = occupied_nodes_global_indices.size();
    if( num_of_rows_to_remove != 0 ){
	PetscInt *rows_global_indices = &occupied_nodes_global_indices[0];
	PetscScalar diag = 1.0;
	/* Approx solution and RHS at zeroed rows */
	inner_region.phi_inside_region = NULL;
	inner_region.rhs_inside_region = NULL;

	// looks like setting phi_inside_region and
	// rhs_inside_region has no effect	
	
	// todo: separate function
	// std::string vec_name = "Phi inside " + inner_region.name;
	// alloc_petsc_vector( &inner_region.phi_inside_region,
	// 		    (nx-2) * (ny-2) * (nz-2),
	// 		    vec_name.c_str() );
	// VecSet( inner_region.phi_inside_region, inner_region.potential );    
	// ierr = VecAssemblyBegin( inner_region.phi_inside_region ); CHKERRXX( ierr );
	// ierr = VecAssemblyEnd( inner_region.phi_inside_region ); CHKERRXX( ierr );

	// todo: separate function
	// vec_name = "RHS inside " + inner_region.name;
	// PetscScalar charge_density_inside_conductor = 0.0;
	// alloc_petsc_vector( &inner_region.rhs_inside_region,
	// 		    (nx-2) * (ny-2) * (nz-2),
	// 		    vec_name.c_str() );
	// VecSet( inner_region.rhs_inside_region, charge_density_inside_conductor );    
	// ierr = VecAssemblyBegin( inner_region.rhs_inside_region ); CHKERRXX( ierr );
	// ierr = VecAssemblyEnd( inner_region.rhs_inside_region ); CHKERRXX( ierr );
	
	ierr = MatZeroRows( *A, num_of_rows_to_remove, rows_global_indices,
			    diag,
			    inner_region.phi_inside_region,
			    inner_region.rhs_inside_region); CHKERRXX( ierr );

	// VecDestroy for phi_inside_region and rhs_inside_region
	// should be called in inner_region destructor.
    }
}

void Field_solver::modify_equation_near_object_boundaries( Mat *A,
							   int nx, int ny, int nz,
							   double dx, double dy, double dz,
							   Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions )
	modify_equation_near_object_boundaries( A, nx, ny, nz, dx, dy, dz, reg );
}


void Field_solver::modify_equation_near_object_boundaries( Mat *A,
							   int nx, int ny, int nz,
							   double dx, double dy, double dz,
							   Inner_region &inner_region )
{
    PetscErrorCode ierr;
    int max_possible_neighbours = 6; // in 3d case; todo: make max_possible_nbr a property of Node_reference
    std::vector<PetscScalar> zeroes( max_possible_neighbours, 0.0 );
    
    for( auto &node : inner_region.near_boundary_nodes_not_at_domain_edge ){
	PetscInt modify_single_row = 1;
	PetscInt row_to_modify = node_global_index_in_matrix( node, nx, ny, nz );
	std::vector<PetscInt> cols_to_modify =
	    adjacent_nodes_not_at_domain_edge_and_inside_inner_region( node, inner_region,
								       nx, ny, nz, dx, dy, dz );
	PetscInt n_of_cols_to_modify = cols_to_modify.size();
	
	if( n_of_cols_to_modify != 0 ){
	    PetscInt *col_indices = &cols_to_modify[0];
	    ierr = MatSetValues( *A,
				 modify_single_row, &row_to_modify,
				 n_of_cols_to_modify, col_indices,
				 &zeroes[0], INSERT_VALUES );
	    CHKERRXX( ierr );	
	}
    }
    
    ierr = MatAssemblyBegin( *A, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *A, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
}


std::vector<PetscInt> Field_solver::adjacent_nodes_not_at_domain_edge_and_inside_inner_region(
    Node_reference &node,
    Inner_region &inner_region,
    int nx, int ny, int nz,
    double dx, double dy, double dz )
{
    int max_possible_neighbours = 6; // in 3d case; todo: make max_possible_nbr a property of Node_reference
    std::vector<PetscInt> resulting_global_indices;
    resulting_global_indices.reserve( max_possible_neighbours );
    for( auto &adj_node : node.adjacent_nodes() ){
	if( !adj_node.at_domain_edge( nx, ny, nz ) &&
	    inner_region.check_if_node_inside( adj_node, dx, dy, dz ) ){
	    resulting_global_indices.push_back( node_global_index_in_matrix( adj_node, nx, ny, nz ) );
	}
    }
    return resulting_global_indices;
}


void Field_solver::construct_d2dx2_in_3d( Mat *d2dx2_3d,
			        int nx, int ny, int nz,
			        PetscInt rstart, PetscInt rend )
{
    PetscErrorCode ierr;
    //int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    //int ncol = nrow;
    int at_boundary_pattern_size = 2;
    int no_boundaries_pattern_size = 3;
    PetscScalar at_left_boundary_pattern[at_boundary_pattern_size];
    PetscScalar no_boundaries_pattern[no_boundaries_pattern_size];
    PetscScalar at_right_boundary_pattern[at_boundary_pattern_size];
    PetscInt cols[ no_boundaries_pattern_size ]; 
    at_right_boundary_pattern[0] = -2.0;
    at_right_boundary_pattern[1] = 1.0;
    no_boundaries_pattern[0] = 1.0;
    no_boundaries_pattern[1] = -2.0;
    no_boundaries_pattern[2] = 1.0;
    at_left_boundary_pattern[0] = 1.0;
    at_left_boundary_pattern[1] = -2.0;

    int i, j, k;
    for( int row_idx = rstart; row_idx < rend; row_idx++ ) {
	global_index_in_matrix_to_node_ijk( row_idx, &i, &j, &k, nx, ny, nz );
	if ( i == 1 ) {
	    // right boundary
	    cols[0] = row_idx;
	    cols[1] = row_idx + 1;
	    ierr = MatSetValues( *d2dx2_3d,
				 1, &row_idx,
				 at_boundary_pattern_size, cols,
				 at_right_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else if ( i == nx - 2 ) {
	    // left boundary
	    cols[0] = row_idx - 1;
	    cols[1] = row_idx;
	    ierr = MatSetValues( *d2dx2_3d,
				 1, &row_idx,
				 at_boundary_pattern_size, cols,
				 at_left_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else {
	    // center
	    cols[0] = row_idx - 1;
	    cols[1] = row_idx;
	    cols[2] = row_idx + 1;
	    ierr = MatSetValues( *d2dx2_3d,
				 1, &row_idx,
				 no_boundaries_pattern_size, cols, 
				 no_boundaries_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	}
	//printf( "d2dx2 loop: i = %d \n", i );
    }		
    
    ierr = MatAssemblyBegin( *d2dx2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *d2dx2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    return;
}


void Field_solver::construct_d2dy2_in_3d( Mat *d2dy2_3d,
			        int nx, int ny, int nz,
			        PetscInt rstart, PetscInt rend )
{
    PetscErrorCode ierr;
    //int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    //int ncol = nrow;
    int at_boundary_pattern_size = 2;
    int no_boundaries_pattern_size = 3;
    PetscScalar at_top_boundary_pattern[at_boundary_pattern_size];
    PetscScalar no_boundaries_pattern[no_boundaries_pattern_size];
    PetscScalar at_bottom_boundary_pattern[at_boundary_pattern_size];
    PetscInt cols[ no_boundaries_pattern_size ];
    at_bottom_boundary_pattern[0] = -2.0;
    at_bottom_boundary_pattern[1] = 1.0;
    no_boundaries_pattern[0] = 1.0;
    no_boundaries_pattern[1] = -2.0;
    no_boundaries_pattern[2] = 1.0;
    at_top_boundary_pattern[0] = 1.0;
    at_top_boundary_pattern[1] = -2.0;
  
    int i, j, k;
    for( int row_idx = rstart; row_idx < rend; row_idx++ ) {
	global_index_in_matrix_to_node_ijk( row_idx, &i, &j, &k, nx, ny, nz );
	if ( j == 1 ) {
	    // bottom boundary
	    cols[0] = row_idx;
	    cols[1] = row_idx + ( nx - 2 );
	    ierr = MatSetValues( *d2dy2_3d,
				 1, &row_idx,
				 at_boundary_pattern_size, cols,
				 at_bottom_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else if ( j == ny - 2 ) {
	    // top boundary
	    cols[0] = row_idx - ( nx - 2 );
	    cols[1] = row_idx;
	    ierr = MatSetValues( *d2dy2_3d,
				 1, &row_idx,
				 at_boundary_pattern_size, cols,
				 at_top_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else {
	    // center
	    cols[0] = row_idx - ( nx - 2 );
	    cols[1] = row_idx;
	    cols[2] = row_idx + ( nx - 2 );
	    ierr = MatSetValues( *d2dy2_3d,
				 1, &row_idx,
				 no_boundaries_pattern_size, cols, 
				 no_boundaries_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	}
	//printf( "d2dx2 loop: i = %d \n", i );
    }		
    
    ierr = MatAssemblyBegin( *d2dy2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *d2dy2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    return;
}

void Field_solver::construct_d2dz2_in_3d( Mat *d2dz2_3d, int nx, int ny, int nz,
					  PetscInt rstart, PetscInt rend )
{
    PetscErrorCode ierr;    
    //int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    //int ncol = nrow;
    const int at_boundary_pattern_size = 2;
    const int no_boundaries_pattern_size = 3;
    PetscScalar at_near_boundary_pattern[at_boundary_pattern_size];
    PetscScalar no_boundaries_pattern[no_boundaries_pattern_size];
    PetscScalar at_far_boundary_pattern[at_boundary_pattern_size];
    PetscInt cols[ no_boundaries_pattern_size ]; 
    at_near_boundary_pattern[0] = -2.0;
    at_near_boundary_pattern[1] = 1.0;
    no_boundaries_pattern[0] = 1.0;
    no_boundaries_pattern[1] = -2.0;
    no_boundaries_pattern[2] = 1.0;
    at_far_boundary_pattern[0] = 1.0;
    at_far_boundary_pattern[1] = -2.0;
  
    for( int i = rstart; i < rend; i++ ) {	
	if ( i < ( nx - 2 ) * ( ny - 2 ) ) {
	    // near boundary
	    cols[0] = i;
	    cols[1] = i + ( nx - 2 ) * ( ny - 2 );
	    ierr = MatSetValues( *d2dz2_3d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_near_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else if ( i >= ( nx - 2 ) * ( ny - 2 ) * ( nz - 3 ) ) {
	    // far boundary
	    cols[0] = i - ( nx - 2 ) * ( ny - 2 );
	    cols[1] = i;
	    ierr = MatSetValues( *d2dz2_3d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_far_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else {
	    // center
	    cols[0] = i - ( nx - 2 ) * ( ny - 2 );
	    cols[1] = i;
	    cols[2] = i + ( nx - 2 ) * ( ny - 2 );
	    ierr = MatSetValues( *d2dz2_3d,
				 1, &i,
				 no_boundaries_pattern_size, cols, 
				 no_boundaries_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	}
    }
    
    ierr = MatAssemblyBegin( *d2dz2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *d2dz2_3d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    return;
}


void Field_solver::create_solver_and_preconditioner( KSP *ksp, PC *pc, Mat *A )
{
    PetscReal rtol = 1.e-12;
    // Default.     
    // Possible to specify from command line using '-ksp_rtol' option.
    
    PetscErrorCode ierr;
    ierr = KSPCreate( PETSC_COMM_WORLD, ksp ); CHKERRXX(ierr);
    //ierr = KSPSetOperators( *ksp, *A, *A, DIFFERENT_NONZERO_PATTERN ); CHKERRXX(ierr);
    ierr = KSPSetOperators( *ksp, *A, *A ); CHKERRXX(ierr);
    ierr = KSPGetPC( *ksp, pc ); CHKERRXX(ierr);
    ierr = PCSetType( *pc, PCGAMG ); CHKERRXX(ierr);
    ierr = KSPSetType( *ksp, KSPGMRES ); CHKERRXX(ierr);
    ierr = KSPSetTolerances( *ksp, rtol,
			     PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
    ierr = KSPSetFromOptions( *ksp ); CHKERRXX(ierr);    
    ierr = KSPSetInitialGuessNonzero( *ksp, PETSC_TRUE ); CHKERRXX( ierr );

    // For test purposes
    //ierr = KSPSetInitialGuessNonzero( *ksp, PETSC_FALSE ); CHKERRXX( ierr );

    ierr = KSPSetUp( *ksp ); CHKERRXX(ierr);
    return;
}

void Field_solver::eval_potential( Spatial_mesh &spat_mesh,
				   Inner_regions_manager &inner_regions )
{
    solve_poisson_eqn( spat_mesh, inner_regions );
}

void Field_solver::solve_poisson_eqn( Spatial_mesh &spat_mesh,
				      Inner_regions_manager &inner_regions )
{
    PetscErrorCode ierr;

    init_rhs_vector( spat_mesh, inner_regions );    
    ierr = KSPSolve( ksp, rhs, phi_vec); CHKERRXX( ierr );
    
    // This should be done in 'cross_out_nodes_occupied_by_objects' by
    // MatZeroRows function but it seems it doesn't work
    set_solution_at_nodes_of_inner_regions( spat_mesh, inner_regions );
    
    transfer_solution_to_spat_mesh( spat_mesh );
    
    return;
}

void Field_solver::init_rhs_vector( Spatial_mesh &spat_mesh,
				    Inner_regions_manager &inner_regions )
{
    init_rhs_vector_in_full_domain( spat_mesh );

    // This should be done in 'cross_out_nodes_occupied_by_objects' by
    // MatZeroRows function but it seems it doesn't work
    set_rhs_at_nodes_occupied_by_objects( spat_mesh, inner_regions );

    modify_rhs_near_object_boundaries( spat_mesh, inner_regions );
}

void Field_solver::init_rhs_vector_in_full_domain( Spatial_mesh &spat_mesh )
{
    PetscErrorCode ierr;
    
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    //int nrow = (nx-2)*(ny-2);
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
		ierr = VecSetValue( rhs,
				    node_ijk_to_global_index_in_matrix( i, j, k, nx, ny, nz ),
				    rhs_at_node, INSERT_VALUES );
		CHKERRXX( ierr );
	    }
	}
    }
    
    ierr = VecAssemblyBegin( rhs ); CHKERRXX( ierr );
    ierr = VecAssemblyEnd( rhs ); CHKERRXX( ierr );
    
    return;
}

void Field_solver::set_rhs_at_nodes_occupied_by_objects( Spatial_mesh &spat_mesh,
							 Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions )
	set_rhs_at_nodes_occupied_by_objects( spat_mesh, reg );
}

void Field_solver::set_rhs_at_nodes_occupied_by_objects( Spatial_mesh &spat_mesh,
							 Inner_region &inner_region )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;

    std::vector<PetscInt> indices_of_inner_nodes_not_at_domain_edge;
    indices_of_inner_nodes_not_at_domain_edge =
	list_of_nodes_global_indices_in_matrix( inner_region.inner_nodes_not_at_domain_edge, nx, ny, nz );

    PetscErrorCode ierr;
    PetscInt num_of_elements = indices_of_inner_nodes_not_at_domain_edge.size();
    if( num_of_elements != 0 ){
	PetscInt *global_indices = &indices_of_inner_nodes_not_at_domain_edge[0];
	std::vector<PetscScalar> zeroes( num_of_elements, 0.0 );
    
	ierr = VecSetValues( rhs, num_of_elements, global_indices, &zeroes[0], INSERT_VALUES );
	CHKERRXX( ierr );

	ierr = VecAssemblyBegin( rhs ); CHKERRXX( ierr );
	ierr = VecAssemblyEnd( rhs ); CHKERRXX( ierr );
    }
}


void Field_solver::modify_rhs_near_object_boundaries( Spatial_mesh &spat_mesh,
						      Inner_regions_manager &inner_regions )
{
    for( auto &reg : inner_regions.regions )
	modify_rhs_near_object_boundaries( spat_mesh, reg );
}


void Field_solver::modify_rhs_near_object_boundaries( Spatial_mesh &spat_mesh,
						      Inner_region &inner_region )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    //int nrow = (nx-2)*(ny-2);
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;

    PetscErrorCode ierr;    
    std::vector<PetscInt> indices_of_nodes_near_boundaries;
    std::vector<PetscScalar> rhs_modification_for_nodes_near_boundaries;

    indicies_of_near_boundary_nodes_and_rhs_modifications(
	indices_of_nodes_near_boundaries,
	rhs_modification_for_nodes_near_boundaries,
	nx, ny, nz,
	dx, dy, dz,
	inner_region );
    
    PetscInt number_of_elements = indices_of_nodes_near_boundaries.size();
    if( number_of_elements != 0 ){
	PetscInt *indices = &indices_of_nodes_near_boundaries[0];
	PetscScalar *values = &rhs_modification_for_nodes_near_boundaries[0];
	// ADD_VALUES gathers values from all processes.
	// Therefore, only a single process
	// should be responsible for calculation of rhs_modification
	// for a given node.
	ierr = VecSetValues( rhs, number_of_elements,
			     indices, values, ADD_VALUES ); CHKERRXX( ierr );
	CHKERRXX( ierr );
	
	ierr = VecAssemblyBegin( rhs ); CHKERRXX( ierr );
	ierr = VecAssemblyEnd( rhs ); CHKERRXX( ierr );
    }
}


void Field_solver::indicies_of_near_boundary_nodes_and_rhs_modifications(
    std::vector<PetscInt> &indices_of_nodes_near_boundaries,
    std::vector<PetscScalar> &rhs_modification_for_nodes_near_boundaries,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    Inner_region &inner_region )
{
    int max_possible_nodes_where_to_modify_rhs = inner_region.near_boundary_nodes_not_at_domain_edge.size();
    indices_of_nodes_near_boundaries.reserve( max_possible_nodes_where_to_modify_rhs );
    rhs_modification_for_nodes_near_boundaries.reserve( max_possible_nodes_where_to_modify_rhs );

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( PETSC_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_process_rank );
    
    for( auto &node : inner_region.near_boundary_nodes_not_at_domain_edge ){
	PetscScalar rhs_mod = 0.0;
	// todo: parallelize instead of making one process
	// to do all the work. 
	if( mpi_process_rank == 0 ){
	    for( auto &adj_node : node.adjacent_nodes() ){
		// possible todo: separate function for rhs_mod evaluation?
		if( !adj_node.at_domain_edge( nx, ny, nz ) &&
		    inner_region.check_if_node_inside( adj_node, dx, dy, dz ) ){
		    if( adj_node.left_from( node ) ) {
			rhs_mod += -inner_region.potential * dy * dy * dz * dz;
		    } else if( adj_node.right_from( node ) ) {
			rhs_mod += -inner_region.potential * dy * dy * dz * dz;
		    } else if( adj_node.top_from( node ) ) {
			rhs_mod += -inner_region.potential * dx * dx * dz * dz;
		    } else if( adj_node.bottom_from( node ) ) {
			rhs_mod += -inner_region.potential * dx * dx * dz * dz;
		    } else if( adj_node.near_from( node ) ) {
			rhs_mod += -inner_region.potential * dx * dx * dy * dy;
		    } else if( adj_node.far_from( node ) ) {
			rhs_mod += -inner_region.potential * dx * dx * dy * dy;
		    }
		}
	    }
	}
	indices_of_nodes_near_boundaries.push_back(
	    node_global_index_in_matrix( node, nx, ny, nz ) );
	rhs_modification_for_nodes_near_boundaries.push_back( rhs_mod );
    }
}

void Field_solver::set_solution_at_nodes_of_inner_regions( Spatial_mesh &spat_mesh,
							   Inner_regions_manager &inner_regions ) 
{
    for( auto &reg : inner_regions.regions )
	set_solution_at_nodes_of_inner_regions( spat_mesh, reg );
}

void Field_solver::set_solution_at_nodes_of_inner_regions( Spatial_mesh &spat_mesh,
							   Inner_region &inner_region )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    
    std::vector<int> occupied_nodes_global_indices =
	list_of_nodes_global_indices_in_matrix( inner_region.inner_nodes_not_at_domain_edge, nx, ny, nz );
    
    PetscErrorCode ierr;
    PetscInt num_of_elements_to_set = occupied_nodes_global_indices.size();
    if( num_of_elements_to_set != 0 ){
	std::vector<PetscScalar> phi_inside_region(num_of_elements_to_set);
	std::fill( phi_inside_region.begin(), phi_inside_region.end(), inner_region.potential );
	
	PetscInt *global_indices = &occupied_nodes_global_indices[0];
	PetscScalar *values = &phi_inside_region[0];
	
	ierr = VecSetValues( phi_vec, num_of_elements_to_set, global_indices,
			     values, INSERT_VALUES); CHKERRXX( ierr );
	ierr = VecAssemblyBegin( phi_vec ); CHKERRXX( ierr );
	ierr = VecAssemblyEnd( phi_vec ); CHKERRXX( ierr );
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


void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    int recieved_rstart, recieved_rend, recieved_nlocal;
    double *local_phi_values;

    int mpi_n_of_proc, mpi_process_rank;
    MPI_Comm_size( PETSC_COMM_WORLD, &mpi_n_of_proc );
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_process_rank );

    MPI_Barrier( PETSC_COMM_WORLD );
    for( int proc = 0; proc < mpi_n_of_proc; proc++ ){
	bcast_phi_array_size( &recieved_rstart, &recieved_rend, &recieved_nlocal, proc, mpi_process_rank );
	allocate_and_populate_phi_array( &local_phi_values, recieved_nlocal, proc, mpi_process_rank );
	transfer_from_phi_array_to_spat_mesh_potential( local_phi_values, recieved_rstart, recieved_rend, spat_mesh );
	deallocate_phi_array( local_phi_values, proc, mpi_process_rank );
    }
}

void Field_solver::bcast_phi_array_size( int *recieved_rstart, int *recieved_rend, int *recieved_nlocal,
					 int proc, int mpi_process_rank )
{
    if( proc == mpi_process_rank ){
	*recieved_rstart = rstart;
	*recieved_rend = rend;
    }
    MPI_Bcast( recieved_rstart, 1, MPI_INT, proc, PETSC_COMM_WORLD );
    MPI_Bcast( recieved_rend, 1, MPI_INT, proc, PETSC_COMM_WORLD );
    *recieved_nlocal = *recieved_rend - *recieved_rstart;
    return;
}

void Field_solver::allocate_and_populate_phi_array( double **local_phi_values, int recieved_nlocal,
						    int proc, int mpi_process_rank )
{
    PetscErrorCode ierr;
    if( proc == mpi_process_rank ){
	ierr = VecGetArray( phi_vec, local_phi_values ); CHKERRXX( ierr );
    } else {
	*local_phi_values = new double [recieved_nlocal];
    }
    MPI_Bcast( *local_phi_values, recieved_nlocal, MPI_DOUBLE, proc, PETSC_COMM_WORLD );
}

void Field_solver::transfer_from_phi_array_to_spat_mesh_potential( double *local_phi_values,
								   int recieved_rstart, int recieved_rend,
								   Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    int i,j,k;

    for( int global_index = recieved_rstart; global_index < recieved_rend; global_index++ ){
	global_index_in_matrix_to_node_ijk( global_index,
					    &i, &j, &k,
					    nx, ny, nz );
	spat_mesh.potential[i][j][k] = local_phi_values[ global_index - recieved_rstart ];
    }
    
}

void Field_solver::deallocate_phi_array( double *local_phi_values, int proc, int mpi_process_rank )
{
    PetscErrorCode ierr;

    if( proc == mpi_process_rank ){
	ierr = VecRestoreArray( phi_vec, &local_phi_values ); CHKERRXX( ierr );
    } else {
	delete[] local_phi_values;
    }		    
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


Field_solver::~Field_solver()
{    
    PetscErrorCode ierr;
    ierr = VecDestroy( &phi_vec ); CHKERRXX( ierr );
    ierr = VecDestroy( &rhs ); CHKERRXX( ierr );
    ierr = MatDestroy( &A ); CHKERRXX( ierr );
    ierr = KSPDestroy( &ksp ); CHKERRXX( ierr );
}
