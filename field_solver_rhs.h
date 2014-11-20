#ifndef _FIELD_SOLVER_RHS_H_
#define _FIELD_SOLVER_RHS_H_

#define _USE_MATH_DEFINES
#include <cmath>

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include "particle.h"
#include "particle_source.h"

template <int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
    RightHandSide( Particle_sources &particle_sources  ) : 
	dealii::Function<dim>(), particle_sources( particle_sources ) {}

    virtual double value (const dealii::Point<dim>   &p,
			  const unsigned int  component = 0) const;

private:
    Particle_sources &particle_sources;

};

template <int dim>
double RightHandSide<dim>::value (const dealii::Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    double rhs = 0;
    for( auto &src : particle_sources.sources ) {
	for( auto &particle : src.particles ) {
	    if( particle.point_inside( p ) )
		rhs += -4.0 * M_PI * particle.charge;
	}
    }
    return rhs;
}

#endif /* _FIELD_SOLVER_RHS_H_ */
