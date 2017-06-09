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

void solve_poisson_cuda( double *b ){


   typedef cusp::array1d<ValueType,MemorySpace> Array;
   typedef typename Array::const_view ConstArrayView;
  // Allocate a array of size 2 in "host" memory
  Array a(8);

for (int i =0; i<8; i++){
	a[i] = b[i]/(0.4*0.4*0.4*0.4);
}


/*
double * device_a;
cudaMalloc(&device_a, 27*sizeof(double));

cudaMemcpy(device_a,b,27*sizeof(double),cudaMemcpyHostToDevice);

thrust::device_ptr<int> wrapped_device_a(device_a);
*/
    cusp::hyb_matrix<int, ValueType, MemorySpace> A;
    cusp::gallery::poisson7pt(A, 2,2,2);

    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);   



    //set stopping criteria:
    //iteration_limit    = 100
    //relative_tolerance = 1e-3
    //absolute_tolerance = 0
    //verbose            = true
    cusp::monitor<ValueType> monitor(a, 1000, 1e-3, 0, true);

    // set preconditioner (identity)
    cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_rows);
 //cusp::precond::aggregation::smoothed_aggregation<int, ValueType, MemorySpace> M(A);
    cusp::krylov::cg(A, x, a, monitor, M);

   // cudaFree(device_a);
	
}


