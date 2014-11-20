#ifndef _FIELD_SOLVER_BV_H_
#define _FIELD_SOLVER_BV_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include "domain_geometry.h"
#include "config.h"

template <int dim>
class BoundaryValues : public dealii::Function<dim>
{
public:
    BoundaryValues( Config &conf, Domain_geometry &domain_geometry ) : 
	dealii::Function<dim>(), conf( conf ), domain_geometry( domain_geometry ) {}

    virtual double value (const dealii::Point<dim>   &p,
			  const unsigned int  component = 0) const;
    
private:
    Config &conf;
    Domain_geometry &domain_geometry;
};

template <int dim>
double BoundaryValues<dim>::value (const dealii::Point<dim> &p,
                                   const unsigned int /*component*/) const
{
    //double return_value = 0;
    if ( domain_geometry.at_left_boundary( p ) ) {
	return conf.boundary_config_part.boundary_phi_left;
    } else if ( domain_geometry.at_top_boundary( p ) ) {
	return conf.boundary_config_part.boundary_phi_top;
    } else if ( domain_geometry.at_right_boundary( p ) ) {
	return conf.boundary_config_part.boundary_phi_right;
    } else if ( domain_geometry.at_bottom_boundary( p ) ) {
	return conf.boundary_config_part.boundary_phi_bottom;	    
    } else {
	std::cout << "Point at unknown boundary. Aborting.";
	exit( EXIT_FAILURE );
    }
}

#endif /* _FIELD_SOLVER_BV_H_ */
