#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/print.h>
#include <time.h>
#include <stdio.h>
#include <cusp/blas/blas.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float ValueType;

__global__ void kernel( )
{
	
}

void as_kernel( ){
	kernel<<<1,1>>>(); 	
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, ValueType, MemorySpace> A;

    // create a 2d Poisson problem on a 10x10 mesh
    cusp::gallery::poisson7pt(A, 100,100,100);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
	cusp::array1d<ValueType, MemorySpace> y(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-3
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<ValueType> monitor(b, 200000, 1e-6, 0, true);

    // set preconditioner (identity)
    cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_rows);
	struct tm *local;
	time_t t;
	t = time(NULL);
	local = localtime(&t);
	printf("Local time and date: %s", asctime(local));
	int i;
	for (i=0; i<10; i=i+1) {
    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b, monitor, M);
	
    cusp::blas::axpy(x, x, 0);
	}
	t = time(NULL);
	local = localtime(&t);
	printf("Local time and date: %s", asctime(local));
}


