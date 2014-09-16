#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include "spatial_mesh.h"

class Field_solver {
  public:
    Field_solver() {};
    void eval_potential( Spatial_mesh *spat_mesh );
    void eval_fields_from_potential( Spatial_mesh *spat_mesh );
    virtual ~Field_solver() {};
  private:
    // Solve potential
    void solve_poisson_eqn( Spatial_mesh *spat_mesh );
    double **poisson_init_rhs( Spatial_mesh *spat_mesh );
    void rowmajor_to_colmajor( double **c, double *fortran, int dim1, int dim2 );
    void colmajor_to_rowmajor( double *fortran, double **c, int dim1, int dim2 );
    void poisson_free_rhs( double **rhs, int nrow, int ncol );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

extern "C" void hwscrt_( double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, int *, double *, double *,
			 double *, double *, int *, double *, int *, double * );



#endif /* _FIELD_SOLVER_H_ */
