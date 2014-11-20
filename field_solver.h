/* ---------------------------------------------------------------------
 * $Id: step-4.cc 30526 2013-08-29 20:06:27Z felix.gruber $
 *
 * Copyright (C) 1999 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */

#ifndef _FIELD_SOLVER_H_
#define _FIELD_SOLVER_H_

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

#include "field_solver_rhs.h"
#include "field_solver_bv.h"

#include "domain_geometry.h"
#include "config.h"
#include "particle.h"
#include "particle_source.h"

using namespace dealii;

template <int dim>
class Field_solver
{
public:
    Field_solver( Config &conf, 
		  Domain_geometry &domain_geometry,
		  Particle_sources &particle_sources );
    void eval_potential_and_fields();    
    Point<dim> force_on_particle( Particle<dim> &p );
    virtual ~Field_solver() {};

private:
    void make_grid( Config &conf, Domain_geometry &domain_geometry );
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;

    void solution_gradient_at_vertices_near_point( 
	const Point<dim> point,
	std::vector< Tensor< 1, dim > > &solution_gradient_at_vertices,
	std::vector< double > &vertices_weights );


    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    RightHandSide<dim>   right_hand_side;
    BoundaryValues<dim>  boundary_values_function;

    Vector<double>       solution;
    Vector<double>       system_rhs;
};



template <int dim>
Field_solver<dim>::Field_solver( Config &conf, 
				 Domain_geometry &domain_geometry,
				 Particle_sources &particle_sources ) :
    fe( 2 ),
    dof_handler( triangulation ),
    right_hand_side( particle_sources ),
    boundary_values_function( conf, domain_geometry )
{
    make_grid( conf, domain_geometry );    
    setup_system();
}



template <int dim>
void Field_solver<dim>::make_grid( Config &conf, Domain_geometry &domain_geometry )
{
  GridGenerator::hyper_rectangle( triangulation, 
				  domain_geometry.bottom_left,
				  domain_geometry.top_right );
  triangulation.refine_global( 5 );

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}




template <int dim>
void Field_solver<dim>::setup_system()
{
  dof_handler.distribute_dofs( fe );

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  CompressedSparsityPattern c_sparsity( dof_handler.n_dofs() );
  DoFTools::make_sparsity_pattern( dof_handler, c_sparsity );
  sparsity_pattern.copy_from( c_sparsity );

  system_matrix.reinit( sparsity_pattern );

  solution.reinit( dof_handler.n_dofs() );
  system_rhs.reinit( dof_handler.n_dofs() );
}




template <int dim>
void Field_solver<dim>::eval_potential_and_fields()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  assemble_system();
  solve();
  output_results();
}




template <int dim>
void Field_solver<dim>::assemble_system()
{
  QGauss<dim>  quadrature_formula(2);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                   fe_values.shape_grad (j, q_point) *
                                   fe_values.JxW (q_point));

            cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                            right_hand_side.value (fe_values.quadrature_point (q_point)) *
                            fe_values.JxW (q_point));
          }

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }


  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values( dof_handler,
                                            0,
                                            boundary_values_function,
                                            boundary_values );
  MatrixTools::apply_boundary_values( boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs );
}



template <int dim>
void Field_solver<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}



template <int dim>
void Field_solver<dim>::output_results () const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::ofstream output (dim == 2 ?
                        "solution-2d.plt" :
                        "solution-3d.vtk");

  data_out.write_gnuplot (output);
}




template <int dim>
void Field_solver<dim>::solution_gradient_at_vertices_near_point( 
    const Point<dim> point,
    std::vector< Tensor< 1, dim > > &solution_gradient_at_vertices,
    std::vector< double > &vertices_weights )
{
    double xlen, ylen;

    QTrapez<dim>  only_vertices_quadrature_formula;
    FEValues<dim> fe_values( 
	fe, only_vertices_quadrature_formula, 
	update_gradients | update_quadrature_points );
   
    const unsigned int n_q_points = only_vertices_quadrature_formula.size();
    solution_gradient_at_vertices.resize( n_q_points );
    vertices_weights.resize( n_q_points );
    
    typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
    
    for ( ; cell != endc; ++cell ){
	if( cell->point_inside( point ) ){
	    fe_values.reinit( cell );
	    fe_values.get_function_gradients( solution,
					      solution_gradient_at_vertices );
	    xlen = cell->extent_in_direction( 0 );
	    ylen = cell->extent_in_direction( 1 );

	    for( unsigned int q_point = 0; q_point < n_q_points; ++q_point ){
		Point<dim> dist = fe_values.quadrature_point( q_point ) - point; 
		vertices_weights[ q_point ] = 
		    ( 1 - std::abs( dist[0] ) / xlen ) * 
		    ( 1 - std::abs( dist[1] ) / ylen );		
	    }
	}
    }
    return;
}


template <int dim>
Point<dim> Field_solver<dim>::force_on_particle( Particle<dim> &p )
{
    const Point<dim> pos = p.position;
    std::vector< Tensor< 1, dim > > solution_gradient_at_vertices;
    std::vector< double > vertices_weights;

    solution_gradient_at_vertices_near_point( 
	pos, solution_gradient_at_vertices, vertices_weights );
    
    bool initilize = true; // init with zeros
    Point<dim> field_from_node( initilize ), total_field( initilize ), force( initilize );

    for( int i = 0; i < solution_gradient_at_vertices.size(); i++ ){
	field_from_node = solution_gradient_at_vertices[i] * vertices_weights[i];
	total_field += field_from_node;
    }

    force = total_field * p.charge;
    return force;
}

#endif /* _FIELD_SOLVER_H_ */
