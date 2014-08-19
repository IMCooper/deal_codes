/*
 Solves the real valued vector-wave equation in 3D:
 curl(curl(E)) + E = Js
 
 Js = E.
 
 where the solution is:
 E = (x^2 , y^2 + , z^2).

*/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;

// RHS:
template<int dim>
class RightHandSide : public Function<dim> 
{
public:
    RightHandSide();

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &values) const;    

};

template<int dim>
RightHandSide<dim>::RightHandSide()
:
Function<dim> (dim)
{}

template <int dim>
void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();

    
    for (unsigned int i=0; i<n_points; ++i)
    {
        const Point<dim> &p = points[i];
        // Real:
        value_list[i](0) = p(0)*p(0);
        value_list[i](1) = p(1)*p(1);
        value_list[i](2) = p(2)*p(2);
    }
}
// END RHS

// Dirichlet BCs / exact solution:.
template<int dim>
class ExactSolution : public Function<dim> 
{
public:
    ExactSolution();

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &values) const;

    void curl_value_list(const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> > &value_list);
    

};

template<int dim>
ExactSolution<dim>::ExactSolution()
:
Function<dim> (dim)
{}
// Wave Propagation members:

template <int dim>
void ExactSolution<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();

    for (unsigned int i=0; i<n_points; ++i)
    {
        const Point<dim> &p = points[i];
        // Real:
        value_list[i](0) = p(0)*p(0);
        value_list[i](1) = p(1)*p(1);
        value_list[i](2) = p(2)*p(2);
    }
}
// Additional functions to create Neumann conditions, zero in this case.
template <int dim>
void ExactSolution<dim>::curl_value_list(const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list)
{
  Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
  const unsigned int n_points = points.size();

    
    double exponent;
    for (unsigned int i=0; i<n_points; ++i)
    {
        const Point<dim> &p = points[i];
        // Real:
        value_list[i](0) = 0.0;
        value_list[i](1) = 0.0;
        value_list[i](2) = 0.0;
    }  
}
// END Dirichlet BC


// MAIN MAXWELL CLASS
template <int dim>
class MaxwellProblem
{
public:
    MaxwellProblem (const unsigned int order);
    ~MaxwellProblem ();
    void run ();
private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void process_solution(const unsigned int cycle);

    double calcErrorHcurlNorm();
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FE_Nedelec<dim>            fe;
    ConstraintMatrix     constraints;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;
    
    ConvergenceTable       convergence_table;
    
    // Choose exact solution - used for BCs and error calculation
    ExactSolution<dim> exact_solution;

    // Input paramters (for hp-FE)
    unsigned int p_order;
    unsigned int quad_order;
};

template <int dim>
MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
:
dof_handler (triangulation),
fe(order)
{
    p_order = order;
    quad_order = p_order+2;
    
}
template <int dim>
MaxwellProblem<dim>::~MaxwellProblem ()
{
    dof_handler.clear ();
}


template<int dim>
double MaxwellProblem<dim>::calcErrorHcurlNorm()
{
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
     
     FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values    |  update_gradients |
                              update_quadrature_points  |  update_JxW_values);
     
     // Extractor
     const FEValuesExtractors::Vector E_re(0);
     
     const unsigned int dofs_per_cell = fe.dofs_per_cell;
     
     std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
     
     // storage for exact sol:
     std::vector<Vector<double> > exactsol_list(n_q_points, Vector<double>(fe.n_components()));
     std::vector<Vector<double> > exactcurlsol_list(n_q_points, Vector<double>(fe.n_components()));
     Tensor<1,dim>  exactsol;
     Tensor<1,dim>  exactcurlsol;

     // storage for computed sol:
     std::vector<Tensor<1,dim> > sol(n_q_points);
     Tensor<1,dim> curlsol(n_q_points);
     
     double h_curl_norm=0.0;
     
     unsigned int block_index_i;
     
     typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler.begin_active(),
     endc = dof_handler.end();
     for (; cell!=endc; ++cell)
     {
         fe_values.reinit (cell);
         
         // Store exact values of E and curlE:
         exact_solution.vector_value_list(fe_values.get_quadrature_points(), exactsol_list);
         exact_solution.curl_value_list(fe_values.get_quadrature_points(), exactcurlsol_list);


         // Store computed values at quad points:
         fe_values[E_re].get_function_values(solution, sol);
         
         // Calc values of curlE from fe solution:
         cell->get_dof_indices (local_dof_indices);
         // Loop over quad points to calculate solution:
         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {
             // Split exact solution into real/imaginary parts:
             for (unsigned int component=0;component<dim;component++)
             {
                 exactsol[component] = exactsol_list[q_point][component];
                 exactcurlsol[component] = exactcurlsol_list[q_point][component];
             }
             // Loop over DoFs to calculate curl of solution @ quad point
             curlsol=0.0;
             for (unsigned int i=0; i<dofs_per_cell; ++i)
             {
                 // Construct local curl value @ quad point
                 curlsol += solution(local_dof_indices[i])*fe_values[E_re].curl(i,q_point);
             }
             // Integrate difference at each point:             
             h_curl_norm += ( (exactsol-sol[q_point])*(exactsol-sol[q_point])
                             + (exactcurlsol-curlsol)*(exactcurlsol-curlsol)
                             )*fe_values.JxW(q_point);
         }
     }
    return sqrt(h_curl_norm);
}

template <int dim>
void MaxwellProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::block_wise (dof_handler);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
       
    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 0, constraints);
    
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close ();
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,false);
    
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
    
}
template <int dim>
void MaxwellProblem<dim>::assemble_system ()
{
    QGauss<dim>  quadrature_formula(quad_order);
    QGauss<dim-1> face_quadrature_formula(quad_order);
    
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                     update_normal_vectors | update_JxW_values);
        
    // Extractor
    const FEValuesExtractors::Vector E_re(0);
   
    // Local cell storage:
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    //RHS storage:
    RightHandSide<dim> right_hand_side;
    std::vector<Vector<double> > rhs_value_list(n_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim> rhs_value_vector(dim);

    // loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        
        // Calc RHS values:
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_value_list);        

        // Loop over all element quad points:
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            // store rhs value at this q point & turn into tensor
            for (unsigned int component=0; component<dim; component++)
            {
                rhs_value_vector[component] = rhs_value_list[q_point](component);
            }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                // Construct local matrix:
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    cell_matrix(i,j) += ( (fe_values[E_re].curl(i,q_point)*fe_values[E_re].curl(j,q_point))
                                         + fe_values[E_re].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                        )*fe_values.JxW(q_point);

                }
                // construct local RHS:
                cell_rhs(i) += rhs_value_vector*fe_values[E_re].value(i,q_point)*fe_values.JxW(q_point);
             
            }
        }

        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix, system_rhs);
    }
}
template <int dim>
void MaxwellProblem<dim>::solve ()
{
    /* Direct */
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    
    A_direct.vmult (solution, system_rhs);
    constraints.distribute (solution);
    
}

template<int dim>
void MaxwellProblem<dim>::process_solution(const unsigned int cycle)
{
    
    Vector<double> diff_per_cell(triangulation.n_active_cells());

    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm);

    
    const double L2_error = diff_per_cell.l2_norm();
    
    double Hcurl_error = calcErrorHcurlNorm();
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("L2 Error", L2_error);
    convergence_table.add_value("H(curl) Error", Hcurl_error);
    
}

template <int dim>
void MaxwellProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<4; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':';
        if (cycle == 0)
        {
            /* Cube mesh */
            GridGenerator::hyper_cube (triangulation, -1, 1);
        }
        else
        {
            triangulation.refine_global (1);
        }

        setup_system ();
        assemble_system ();
        solve ();
        process_solution (cycle);
        std::cout << " Done" << std::endl;
    }
    
    convergence_table.set_precision("L2 Error",8);
    convergence_table.set_scientific("L2 Error",true);
    
    convergence_table.set_precision("H(curl) Error",8);
    convergence_table.set_scientific("H(curl) Error",true);
    
    convergence_table.write_text(std::cout);
}
// END MAXWELL CLASS
int main (int argc, char* argv[])
{
    // read order as input via '-p x'
    unsigned int p_order=0;
    if (argc > 0)
    {
        for (int i=1;i<argc;i++)
        {
            if (i+1 != argc)
            {
                std::string input = argv[i];
                if (input == "-p")
                {
                    std::stringstream strValue;
                    strValue << argv[i+1];
                    strValue >> p_order;                        
                }    
            }
        }
    }
    deallog.depth_console (0);
    MaxwellProblem<3> maxwell(p_order);
    maxwell.run ();
    return 0;
}
