#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/print.h>
#include <time.h>
#include <stdio.h>
#include <cusp/blas/blas.h>
#include <cusp/array1d.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef double ValueType;

cusp::hyb_matrix<int, ValueType, MemorySpace> A;
cusp::array1d<ValueType,MemorySpace> local_rhs(1);
cusp::array1d<ValueType,MemorySpace> local_phi(1);

void allocate_matrix_cuda(int nx){
   int nodes = (nx-2)*(nx-2)*(nx-2);
   cusp::gallery::poisson7pt(A, nx-2,nx-2,nx-2);
   local_rhs.resize(nodes);
   local_phi.resize(nodes);
}

void solve_poisson_cuda( double *rhs, double *phi_vec, int nx, double cell_size ){


   int nodes = (nx-2)*(nx-2)*(nx-2);
   for (int i =0; i<nodes; i++){
      local_rhs[i] = -rhs[i]/(cell_size * cell_size * cell_size * cell_size);
      local_phi[i]=phi_vec[i];
   }
   
   
   cusp::array1d<ValueType, MemorySpace> x(local_phi);
   //set stopping criteria:
   //iteration_limit    = 100
   //relative_tolerance = 1e-3
   //absolute_tolerance = 0
   //verbose            = true
   cusp::monitor<ValueType> monitor(local_rhs, 1000, 1e-3, 0, true);

   // set preconditioner (identity)
   cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_rows);
   //cusp::precond::aggregation::smoothed_aggregation<int, ValueType, MemorySpace> M(A);

   cusp::krylov::cg(A, x, local_rhs, monitor, M);
   for (int i = 0; i<nodes; i++){
      phi_vec[i] = x [i];
   } 

	
}


