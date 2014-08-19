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
#include <deal.II/base/convergence_table.h>
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
using namespace dealii;
class Poisson
{
public:
    Poisson (const FiniteElement<2> &fe);
    void run ();
    void process_solution (const unsigned int cycle);
    void write_table ();
//    void set_polynomial_order(int input);
//    void set_refinement_level(int input);
    int quad_order;
    int refinement_level;
    
private:
    
    void make_grid ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results () const;

    Triangulation<2>     triangulation;
//    FE_Q<2>              fe;
    DoFHandler<2>        dof_handler;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;
    SmartPointer<const FiniteElement<2> > fe;
// added:
    ConvergenceTable	convergence_table;
    
};

/*
void set_polynomial_order(int input)
{
    int quad_order=input;
}
void set_refinement_level(int input)
{
    int refinement_level=input;
}
*/

class Solution : public Function<2>
{
public:
  Solution () : Function<2>() {}
  virtual double value (const Point<2>   &p,
                        const unsigned int  component = 0) const;
  virtual Tensor<1,2> gradient (const Point<2>   &p,
                                  const unsigned int  component = 0) const;
};

double Solution::value (const Point<2>   &p,
                             const unsigned int) const
{
  double return_value = -sin(dealii::numbers::PI*p(0))*sin(dealii::numbers::PI*p(1));
  
  return return_value;
}

Tensor<1,2> Solution::gradient (const Point<2>   &p,
                                         const unsigned int) const
  {
    Tensor<1,2> return_value;
    
    return_value[0] = -dealii::numbers::PI*cos(dealii::numbers::PI*p(0))*sin(dealii::numbers::PI*p(1));
    return_value[1] = -dealii::numbers::PI*sin(dealii::numbers::PI*p(0))*cos(dealii::numbers::PI*p(1));
    return return_value;
  }

class RightHandSide : public Function<2>
{
public:
  RightHandSide () : Function<2>() {}
  virtual double value (const Point<2>   &p,
                        const unsigned int  component = 0) const;
};

class BoundaryValues : public Function<2>
{
public:
  BoundaryValues () : Function<2>() {}
  virtual double value (const Point<2>   &p,
                        const unsigned int  component = 0) const;
};

double RightHandSide::value (const Point<2> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = -2*pow(dealii::numbers::PI,2)*sin(dealii::numbers::PI*p(0))*sin(dealii::numbers::PI*p(1));
  return return_value;
}

double BoundaryValues::value (const Point<2> &p,
                                   const unsigned int /*component*/) const
{
  return 0;
}

Poisson::Poisson (const FiniteElement<2> &fe) :
  dof_handler (triangulation),
  fe (&fe)

{}
void Poisson::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (refinement_level);
/* Writes to screen
    std::cout << "Number of active cells: "
    << triangulation.n_active_cells()
    << std::endl;
    std::cout << "Total number of cells: "
    << triangulation.n_cells()
    << std::endl;
 */
}
void Poisson::setup_system ()
{
    dof_handler.distribute_dofs (*fe);
    std::cout << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}
void Poisson::assemble_system ()
{
    QGauss<2>  quadrature_formula(quad_order);
    QGauss<1> face_quadrature_formula(quad_order);
    
    FEValues<2>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<2> fe_face_values (*fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  |
                                      update_normal_vectors | update_JxW_values);
    const unsigned int   dofs_per_cell = fe->dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    const RightHandSide right_hand_side;
    std::vector<double>  rhs_values (n_q_points);
    
    DoFHandler<2>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        cell_matrix = 0;
        cell_rhs = 0;
	
	right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);
	
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                         fe_values.shape_grad (j, q_point) *
                                         fe_values.JxW (q_point));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                cell_rhs(i) += (fe_values.shape_value (i, q_point) *
                                right_hand_side.value (fe_values.quadrature_point (q_point)) *
                                fe_values.JxW (q_point));
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}
void Poisson::solve ()
{
    SolverControl           solver_control (1000, 1e-12);
    SolverCG<>              solver (solver_control);
    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());
}
void Poisson::output_results () const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("solution.gpl");
    data_out.write_gnuplot (output);
    
/* Added
    std::cout << "Solution at (1/3,1/3): "
          << VectorTools::point_value (dof_handler, solution,
                                       Point<2>(1./3, 1./3))
          << std::endl;
  
    std::cout << "Mean value: "
          << VectorTools::compute_mean_value (dof_handler,
                                              QGauss<2>(quad_order),
                                              solution,
                                              0)
          << std::endl;
 */
}

void Poisson::process_solution (const unsigned int cycle)
{
  Vector<float> difference_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (dof_handler,
				     solution,
				     Solution (),
				     difference_per_cell,
				     QGauss<2>(quad_order),
				     VectorTools::L2_norm);
  const double L2_error = difference_per_cell.l2_norm();
  
  VectorTools::integrate_difference (dof_handler,
                                   solution,
                                   Solution (),
                                   difference_per_cell,
                                   QGauss<2>(quad_order),
                                   VectorTools::H1_seminorm);
  const double H1_error = difference_per_cell.l2_norm();
  
  const unsigned int n_active_cells=triangulation.n_active_cells();
  const unsigned int n_dofs=dof_handler.n_dofs();

/*
  std::cout << "Cycle " << cycle << ':'
            << std::endl
            << "   Number of active cells:       "
            << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: "
            << n_dofs
            << std::endl;
*/
  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
    
  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  
}

void Poisson::write_table ()
{
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
}


void Poisson::run ()
{
    make_grid ();
    setup_system();
    assemble_system ();
    solve ();
    output_results ();
}
int main ()
{
    using namespace dealii;
    int polynomial_order=1;
    int refinement_level=3;
    int cycle=1;
//    ConvergenceTable Poisson::convergence_table.add_value;
    FE_Q<2> fe(cycle);
    Poisson laplace_problem(fe);
    
    for (cycle=3; cycle<4; ++cycle)
    {
        FE_Q<2> fe(cycle);
        Poisson laplace_problem(fe);
        //    laplace_problem.set_polynomial_order(polynomial_order);
        //    laplace_problem.set_refinement_level(refinement_level);
        laplace_problem.quad_order=3;
        laplace_problem.refinement_level=refinement_level;
        laplace_problem.run ();
        laplace_problem.process_solution (cycle);
        laplace_problem.write_table ();

    };

    return 0;
}
