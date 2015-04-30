#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <gsl/gsl_linalg.h>
#include "Spatial_mesh.hpp"

class Field_solver {
  public:
    Field_solver( Spatial_mesh &spat_mesh );
    void eval_potential( Spatial_mesh &spat_mesh );
    void eval_fields_from_potential( Spatial_mesh &spat_mesh );
    virtual ~Field_solver();
  private:
    gsl_matrix *a;
    gsl_vector *rhs;
    gsl_vector *phi_vec;
    gsl_permutation *pmt;
    int perm_sign;
    gsl_matrix* construct_equation_matrix( int nx, int ny, double dx, double dy );
    gsl_matrix* construct_d2dx2_in_2d( int nx, int ny );
    gsl_matrix* construct_d2dy2_in_2d( int nx, int ny );
    // Solve potential
    void solve_poisson_eqn( Spatial_mesh &spat_mesh );
    void init_rhs_vector( Spatial_mesh &spat_mesh );
    int kronecker_delta( int i,  int j );
    void transfer_solution_to_spat_mesh( Spatial_mesh &spat_mesh );
    // Eval fields from potential
    double boundary_difference( double phi1, double phi2, double dx );
    double central_difference( double phi1, double phi2, double dx );
};

#endif /* _FIELD_SOLVER_H_ */
