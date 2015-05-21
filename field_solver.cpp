#include "field_solver.h"

Field_solver::Field_solver( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    double dx = spat_mesh.x_cell_size;
    double dy = spat_mesh.y_cell_size;
    double dz = spat_mesh.z_cell_size;
    PetscInt nrows = (nx-2)*(ny-2)*(nz-2);
    PetscInt ncols = nrows;

    PetscInt A_approx_nonzero_per_row = 7;
    PetscBool nonzeroguess = PETSC_FALSE;

    alloc_petsc_vector( &phi_vec, nrows, "Solution" );
    alloc_petsc_vector( &rhs, nrows, "RHS" );
    alloc_petsc_matrix( &A, nrows, ncols, A_approx_nonzero_per_row );
    
    construct_equation_matrix( &A, nx, ny, nz, dx, dy, dz );
    create_solver_and_preconditioner( &ksp, &pc, &A, nonzeroguess, &phi_vec );
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


void Field_solver::alloc_petsc_matrix( Mat *A, PetscInt nrow, PetscInt ncol, PetscInt nonzero_per_row )
{
    PetscErrorCode ierr;

    /* ierr = MatCreate( PETSC_COMM_WORLD, A ); CHKERRXX( ierr ); */
    /* ierr = MatSetSizes( *A, PETSC_DECIDE, PETSC_DECIDE, nrow, ncol ); CHKERRXX( ierr ); */
    /* ierr = MatSetFromOptions( *A ); CHKERRXX( ierr ); */
    /* ierr = MatSeqAIJSetPreallocation( *A, nonzero_per_row, NULL ); CHKERRXX( ierr ); */
    ierr = MatCreateSeqAIJ( PETSC_COMM_WORLD, nrow, ncol,
			    nonzero_per_row, NULL,  A ); CHKERRXX( ierr );
    ierr = MatSetUp( *A ); CHKERRXX( ierr );
    return;
}


void Field_solver::construct_equation_matrix( Mat *A,
					      int nx, int ny, int nz,
					      double dx, double dy, double dz )
{
    PetscErrorCode ierr;
    Mat d2dy2, d2dz2;
    int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    int ncol = nrow;
    PetscInt nonzero_per_row = 7; // approx

    construct_d2dx2_in_3d( A, nx, ny, nz );
    ierr = MatScale( *A, dy * dy * dz * dz ); CHKERRXX( ierr );
    
    alloc_petsc_matrix( &d2dy2, nrow, ncol, nonzero_per_row );
    construct_d2dy2_in_3d( &d2dy2, nx, ny, nz );
    ierr = MatAXPY( *A, dx * dx * dz * dz, d2dy2, DIFFERENT_NONZERO_PATTERN ); CHKERRXX( ierr );
    ierr = MatDestroy( &d2dy2 ); CHKERRXX( ierr );

    alloc_petsc_matrix( &d2dz2, nrow, ncol, nonzero_per_row );
    construct_d2dz2_in_3d( &d2dz2, nx, ny, nz );
    ierr = MatAXPY( *A, dx * dx * dy * dy, d2dz2, DIFFERENT_NONZERO_PATTERN ); CHKERRXX( ierr );
    ierr = MatDestroy( &d2dz2 ); CHKERRXX( ierr );

    return;
}

void Field_solver::construct_d2dx2_in_3d( Mat *d2dx2_3d, int nx, int ny, int nz )
{    
    PetscErrorCode ierr;
    Mat d2dx2_2d;
    int nrow_2d = ( nx - 2 ) * ( ny - 2 );
    int ncol_2d = nrow_2d;
    PetscInt nonzero_per_row = 5; // approx

    alloc_petsc_matrix( &d2dx2_2d, nrow_2d, ncol_2d, nonzero_per_row );
    construct_d2dx2_in_2d( &d2dx2_2d, nx, ny );
    multiply_pattern_along_diagonal( d2dx2_3d, &d2dx2_2d, (nx-2)*(ny-2), nz-2 );
    ierr = MatDestroy( &d2dx2_2d ); CHKERRXX( ierr );

    return;
}

void Field_solver::construct_d2dy2_in_3d( Mat *d2dy2_3d, int nx, int ny, int nz )
{
    PetscErrorCode ierr;
    Mat d2dy2_2d;
    int nrow_2d = ( nx - 2 ) * ( ny - 2 );
    int ncol_2d = nrow_2d;
    PetscInt nonzero_per_row = 5; // approx

    alloc_petsc_matrix( &d2dy2_2d, nrow_2d, ncol_2d, nonzero_per_row );
    construct_d2dy2_in_2d( &d2dy2_2d, nx, ny );
    multiply_pattern_along_diagonal( d2dy2_3d, &d2dy2_2d, (nx-2)*(ny-2), nz-2 );
    ierr = MatDestroy( &d2dy2_2d ); CHKERRXX( ierr );

    return;
}

void Field_solver::construct_d2dz2_in_3d( Mat *d2dz2_3d, int nx, int ny, int nz )
{
    PetscErrorCode ierr;    
    int nrow = ( nx - 2 ) * ( ny - 2 ) * ( nz - 2 );
    //int ncol = nrow;
    int at_boundary_pattern_size = 2;
    int no_boundaries_pattern_size = 3;
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
  
    for( int i = 0; i < nrow; i++ ) {	
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


void Field_solver::multiply_pattern_along_diagonal( Mat *result, Mat *pattern, int pt_size, int n_times )
{
    PetscErrorCode ierr;    
    int mul_nrow = pt_size * n_times;
    //int mul_ncol = mul_nrow;
    int pattern_i;
    // int pattern_j;
    PetscInt pattern_nonzero_cols_number;
    const PetscInt *pattern_nonzero_cols;
    const PetscScalar *pattern_nonzero_vals;
    
    for( int i = 0; i < mul_nrow; i++ ) {
	pattern_i = i%pt_size;
	ierr = MatGetRow( *pattern,
			  pattern_i, &pattern_nonzero_cols_number, &pattern_nonzero_cols,
			  &pattern_nonzero_vals); CHKERRXX( ierr );

	PetscInt result_nonzero_cols[pattern_nonzero_cols_number];

	for( int t = 0; t < pattern_nonzero_cols_number; t++  ){
	    result_nonzero_cols[t] = pattern_nonzero_cols[t] + ( i / pt_size ) * pt_size;
	}

	ierr = MatSetValues( *result,
			     1, &i,
			     pattern_nonzero_cols_number, result_nonzero_cols,
			     pattern_nonzero_vals, INSERT_VALUES); CHKERRXX( ierr );
	MatRestoreRow( *pattern,
		       pattern_i, &pattern_nonzero_cols_number, &pattern_nonzero_cols,
		       &pattern_nonzero_vals ); CHKERRXX( ierr );
    }

    ierr = MatAssemblyBegin( *result, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *result, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    
    return;
}


void Field_solver::construct_d2dx2_in_2d( Mat *d2dx2_2d, int nx, int ny )
{
    PetscErrorCode ierr;
    int nrow = ( nx - 2 ) * ( ny - 2 );
    //int ncol = nrow;
    int at_boundary_pattern_size = 2;
    int no_boundaries_pattern_size = 3;
    PetscScalar at_left_boundary_pattern[at_boundary_pattern_size];
    PetscScalar no_boundaries_pattern[no_boundaries_pattern_size];
    PetscScalar at_right_boundary_pattern[at_boundary_pattern_size];
    PetscInt cols[ no_boundaries_pattern_size ]; 
    at_left_boundary_pattern[0] = -2.0;
    at_left_boundary_pattern[1] = 1.0;
    no_boundaries_pattern[0] = 1.0;
    no_boundaries_pattern[1] = -2.0;
    no_boundaries_pattern[2] = 1.0;
    at_right_boundary_pattern[0] = 1.0;
    at_right_boundary_pattern[1] = -2.0;
    
    for( int i = 0; i < nrow; i++ ) {	
	if ( (i + 1) % ( nx - 2 ) == 1 ) {
	    // left boundary
	    cols[0] = i;
	    cols[1] = i+1;
	    ierr = MatSetValues( *d2dx2_2d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_left_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else if ( (i + 1) % ( nx - 2 ) == 0 ) {
	    // right boundary
	    cols[0] = i-1;
	    cols[1] = i;
	    ierr = MatSetValues( *d2dx2_2d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_right_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else {
	    // center
	    cols[0] = i-1;
	    cols[1] = i;
	    cols[2] = i+1;
	    ierr = MatSetValues( *d2dx2_2d,
				 1, &i,
				 no_boundaries_pattern_size, cols, 
				 no_boundaries_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	}
	//printf( "d2dx2 loop: i = %d \n", i );
    }		
    
    ierr = MatAssemblyBegin( *d2dx2_2d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *d2dx2_2d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    return;
}

void Field_solver::construct_d2dy2_in_2d( Mat *d2dy2_2d, int nx, int ny )
{
    PetscErrorCode ierr;    
    int nrow = ( nx - 2 ) * ( ny - 2 );
    //int ncol = nrow;
    int at_boundary_pattern_size = 2;
    int no_boundaries_pattern_size = 3;
    PetscScalar at_top_boundary_pattern[at_boundary_pattern_size];
    PetscScalar no_boundaries_pattern[no_boundaries_pattern_size];
    PetscScalar at_bottom_boundary_pattern[at_boundary_pattern_size];
    PetscInt cols[ no_boundaries_pattern_size ]; 
    at_top_boundary_pattern[0] = -2.0;
    at_top_boundary_pattern[1] = 1.0;
    no_boundaries_pattern[0] = 1.0;
    no_boundaries_pattern[1] = -2.0;
    no_boundaries_pattern[2] = 1.0;
    at_bottom_boundary_pattern[0] = 1.0;
    at_bottom_boundary_pattern[1] = -2.0;
  
    for( int i = 0; i < nrow; i++ ) {	
	if ( i < nx - 2 ) {
	    // top boundary
	    cols[0] = i;
	    cols[1] = i + ( nx - 2 );
	    ierr = MatSetValues( *d2dy2_2d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_top_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else if ( i >= ( nx - 2 ) * ( ny - 3 ) ) {
	    // bottom boundary
	    cols[0] = i - ( nx - 2 );
	    cols[1] = i;
	    ierr = MatSetValues( *d2dy2_2d,
				 1, &i,
				 at_boundary_pattern_size, cols,
				 at_bottom_boundary_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	} else {
	    // center
	    cols[0] = i - ( nx - 2 );
	    cols[1] = i;
	    cols[2] = i + ( nx - 2 );
	    ierr = MatSetValues( *d2dy2_2d,
				 1, &i,
				 no_boundaries_pattern_size, cols, 
				 no_boundaries_pattern, INSERT_VALUES );
	    CHKERRXX( ierr );
	}
	//printf( "d2dx2 loop: i = %d \n", i );
    }		
    
    ierr = MatAssemblyBegin( *d2dy2_2d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    ierr = MatAssemblyEnd( *d2dy2_2d, MAT_FINAL_ASSEMBLY ); CHKERRXX( ierr );
    return;
}

void Field_solver::create_solver_and_preconditioner( KSP *ksp, PC *pc, Mat *A, PetscBool nonzeroguess, Vec *x )
{
    PetscReal rtol = 1.e-12;
    
    PetscErrorCode ierr;
    ierr = KSPCreate( PETSC_COMM_WORLD, ksp ); CHKERRXX(ierr);
    ierr = KSPSetOperators( *ksp, *A, *A, DIFFERENT_NONZERO_PATTERN ); CHKERRXX(ierr);
    //ierr = KSPSetOperators( *ksp, *A, *A ); CHKERRXX(ierr);
    ierr = KSPGetPC( *ksp, pc ); CHKERRXX(ierr);
    //ierr = PCSetType( *pc, PCJACOBI ); CHKERRXX(ierr);
    ierr = PCSetType( *pc, PCLU ); CHKERRXX(ierr);
    ierr = KSPSetTolerances( *ksp, rtol,
			     PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
    //ierr = KSPSetFromOptions( *ksp ); CHKERRXX(ierr);
    ierr = KSPSetType( *ksp, KSPPREONLY ); CHKERRXX(ierr); 
    
    if ( nonzeroguess ) {
	PetscScalar p = .5;
	ierr = VecSet( *x, p ); CHKERRXX( ierr );
	ierr = KSPSetInitialGuessNonzero( *ksp, PETSC_TRUE ); CHKERRXX( ierr );
    }
    return;
}

void Field_solver::eval_potential( Spatial_mesh &spat_mesh )
{
    solve_poisson_eqn( spat_mesh );
}

void Field_solver::solve_poisson_eqn( Spatial_mesh &spat_mesh )
{
    PetscErrorCode ierr;

    init_rhs_vector( spat_mesh );
    ierr = KSPSolve( ksp, rhs, phi_vec); CHKERRXX( ierr );
    transfer_solution_to_spat_mesh( spat_mesh );
    return;
}

void Field_solver::init_rhs_vector( Spatial_mesh &spat_mesh )
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
    // start processing rho from the near top left corner
    for ( int k = 1; k <= nz-2; k++ ) {
	for ( int j = ny-2; j >= 1; j-- ) { 
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
		// todo: separate function for:
		// (i - 1) + ( ( ny - 2 ) - j ) * (nx-2) + ( nx - 2 ) * ( ny - 2 ) * ( k - 1 )
		ierr = VecSetValue( rhs,
				    (i - 1) + ( ( ny - 2 ) - j ) * (nx-2) + ( nx - 2 ) * ( ny - 2 ) * ( k - 1 ),
				    rhs_at_node, INSERT_VALUES );
		CHKERRXX( ierr );
	    }
	}
    }
    
    ierr = VecAssemblyBegin( rhs ); CHKERRXX( ierr );
    ierr = VecAssemblyEnd( rhs ); CHKERRXX( ierr );
    
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

void Field_solver::transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh )
{
    int nx = spat_mesh.x_n_nodes;
    int ny = spat_mesh.y_n_nodes;
    int nz = spat_mesh.z_n_nodes;
    PetscScalar phi_at_point;
    PetscInt ix;
    PetscErrorCode ierr;
    
    for( int k = 1; k <= nz-2; k++ ){
	for ( int j = ny-2; j >= 1; j-- ) { 
	    for ( int i = 1; i <= nx-2; i++ ) {
		ix = (i - 1) + ( ( ny - 2 ) - j ) * (nx-2) + ( nx - 2 ) * ( ny - 2 ) * ( k - 1 ) ;
		ierr = VecGetValues( phi_vec, 1, &ix, &phi_at_point ); CHKERRXX( ierr );
		spat_mesh.potential[i][j][k] = phi_at_point;
	    }
	}
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
