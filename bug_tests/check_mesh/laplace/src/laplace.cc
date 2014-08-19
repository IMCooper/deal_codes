/*
 Solves the real valued vector-wave equation in 3D:
 curl(curl(E)) + E = Js
 
 Js = E.
 
 where the solution is:
 E = (x^2 , y^2 + , z^2).
 
 with p orthogonal to k, |k| < 1, |p|=1.
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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>


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
        value_list[i](0) = -2.0;
        value_list[i](1) = -2.0;
        value_list[i](2) = -2.0;
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

    void vector_gradient_list(const std::vector< Point< dim > > &     points,
                                              std::vector< std::vector< Tensor< 1, dim > > > &gradients 
                                             ) const;
    

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


void ExactSolution<dim>::vector_gradient_list(const std::vector< Point< dim > > &     points,
                                              std::vector< std::vector< Tensor< 1, dim > > > &gradients 
                                             ) const
{
  Assert(gradients.size() == points.size(), ExcDimensionMismatch(gradients.size(), points.size()));
  const unsigned int n_points = points.size();

    
    double exponent;
    for (unsigned int i=0; i<n_points; ++i)
    {
        const Point<dim> &p = points[i];
        for (unsigned int ii=0;ii<dim;ii++)
        {
            for (unsigned int jj=0;ii<dim;ii++)
            {
                gradients[i][ii][jj] = 0.0;
            }
        }
        // Real:
        gradients[i][0][0] = 2.0;
        gradients[i][1][1] = 2.0;
        gradients[i][2][2] = 2.0;
    }  
}
// END Dirichlet BC


// MAIN MAXWELL CLASS
template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem (const unsigned int order);
    ~LaplaceProblem ();
    void run ();
private:
    class Postprocessor; // DataPostprocessor
    void read_in_mesh (std::string mesh_file);
    void setup_system ();
    void assemble_system ();
    void solve ();
    void process_solution(const unsigned int cycle);
    void output_results_vtk (const unsigned int cycle) const;

    //void mesh_info(const Triangulation<dim> &tria, const std::string        &filename);
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FESystem<dim>          fe;
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
LaplaceProblem<dim>::LaplaceProblem (const unsigned int order)
:
dof_handler (triangulation),
fe (FE_Q<dim>(order),dim),
exact_solution()
{
    p_order = order;
    quad_order = p_order+2;
    
}
template <int dim>
LaplaceProblem<dim>::~LaplaceProblem ()
{
    dof_handler.clear ();
}

// Postprocessor for plotting:
template <int dim>
class LaplaceProblem<dim>::Postprocessor : public DataPostprocessor<dim>
{
public:
    Postprocessor ();
    virtual
    void
    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                       const std::vector<Point<dim> >                  &normals,
                                       const std::vector<Point<dim> >                  &evaluation_points,
                                       std::vector<Vector<double> >                    &computed_quantities) const;
    
    virtual std::vector<std::string> get_names () const;
    virtual
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const;
    virtual UpdateFlags get_needed_update_flags () const;
};

template <int dim>
LaplaceProblem<dim>::Postprocessor::Postprocessor ()
:
DataPostprocessor<dim>()
{}

template <int dim>
std::vector<std::string>
LaplaceProblem<dim>::Postprocessor::get_names() const
{
    std::vector<std::string> solution_names (dim, "E");    
    return solution_names;
}
template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
LaplaceProblem<dim>::Postprocessor::
get_data_component_interpretation () const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    return interpretation;
}
template <int dim>
UpdateFlags
LaplaceProblem<dim>::Postprocessor::get_needed_update_flags() const
{
    return update_values | update_q_points; //update_normal_vectors | | update_gradients | update_hessians
}
template <int dim>
void
LaplaceProblem<dim>::Postprocessor::compute_derived_quantities_vector (const std::vector<Vector<double> >    &uh,
                                                  const std::vector<std::vector<Tensor<1,dim> > >      &duh,
                                                  const std::vector<std::vector<Tensor<2,dim> > >      &dduh,
                                                  const std::vector<Point<dim> >                       &normals,
                                                  const std::vector<Point<dim> >                       &evaluation_points,
                                                  std::vector<Vector<double> >                         &computed_quantities) const
{
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
    Assert (uh[0].size() == dim,                            ExcInternalError());
    
    double temp_scaling = 1.0;
    
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        // Electric field, E:
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](d) = uh[q](d);
        }        
    }
}
// END Postprocessor class.


template <int dim>
void LaplaceProblem<dim>::read_in_mesh (std::string mesh_name)
{
    /* Intended for reading in .ucd files generated by Cubit/Trelis
     * These meshes have material ids (blocks) starting at 1
     * so we take 1 away from all material_id's in the mesh.
     */
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream mesh_file(mesh_name);
    gridin.read_ucd(mesh_file);
    
    // Adjust material_id to start at 0 instead of 1.
    for (typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active();
        cell != triangulation.end();
        ++cell)
    {
        cell->set_material_id(cell->material_id()-1);
    }
}

template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::component_wise (dof_handler);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
    
    DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              exact_solution,
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
void LaplaceProblem<dim>::assemble_system ()
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
    std::vector<Vector<double> > rhs_value_list(n_q_points, Vector<double>(dim));
    Tensor<1,dim> rhs_value_vector(dim);
       
    // Neumann storage
    std::vector<Vector<double> > neumann_value_list(n_face_q_points, Vector<double>(dim));
    Tensor<1,dim> neumann_value_vector(dim);
    Tensor<1,dim> neumann_value(dim);
    Tensor<1,dim> normal_vector;
    
    // loop over all cells:
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell_matrix = 0.0;
        cell_rhs = 0.0;
        
        // Calc RHS values:
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_value_list);        

        // Loop over all element quad points:
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                // Construct local matrix:
                const unsigned int component_i = fe.system_to_component_index(i).first;
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    
                    cell_matrix(i,j) += (fe_values.shape_grad(i,q_point)[component_i]*fe_values.shape_grad(j,q_point)[component_j]      
                                         )*fe_values.JxW(q_point);

                }
                // construct local RHS:
                cell_rhs(i) += rhs_value_list[q_point](component_i)*fe_values.shape_value(i,q_point)*fe_values.JxW(q_point);
             
            }
        }
        // Loop over faces for neumann condition:
//         for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
//         {
//             fe_face_values.reinit (cell, face_number);
//             if (cell->face(face_number)->at_boundary()
//                 &&
//                 (cell->face(face_number)->boundary_indicator() == 1))
//             {
//                 // Store values of (mur^-1)*curl E:
//                 exact_solution.curl_value_list(fe_face_values.get_quadrature_points(), neumann_value_list);
//                 for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
//                 {
//                     for (unsigned int component=0; component<dim; component++)
//                     {
//                         neumann_value_vector[component] = neumann_value_list[q_point](component);
//                         normal_vector[component] = fe_face_values.normal_vector(q_point)(component);
//                     }
//                     cross_product(neumann_value, normal_vector, neumann_value_vector);
//                     for (unsigned int i=0; i<dofs_per_cell; ++i)
//                     {
//                         cell_rhs(i) += -(neumann_value*fe_face_values[E_re].value(i,q_point)*fe_face_values.JxW(q_point));                            
// 
//                     }
//                 }
//             }
//         }
        cell->get_dof_indices (local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix, system_rhs);
    }
}
template <int dim>
void LaplaceProblem<dim>::solve ()
{
    /* Direct */
//     SparseDirectUMFPACK A_direct;
//     A_direct.initialize(system_matrix);
//     
//     A_direct.vmult (solution, system_rhs);
//     constraints.distribute (solution);
    
    /* Iterative */
    SolverControl           solver_control (100000, 1e-6);
    SolverCG<>              cg (solver_control);
    PreconditionJacobi<> preconditioner;
    preconditioner.initialize(system_matrix);
    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);
    constraints.distribute (solution);
    
}
template <int dim>
void LaplaceProblem<dim>::output_results_vtk (const unsigned int cycle) const
{
    
    std::ostringstream filename;
    filename << "solution-" << cycle << ".vtk";
    std::ofstream output (filename.str().c_str());
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);

    Postprocessor postprocessor;
    
    data_out.add_data_vector(solution,postprocessor);
    data_out.build_patches (quad_order);
    data_out.write_vtk (output);
}
template<int dim>
void LaplaceProblem<dim>::process_solution(const unsigned int cycle)
{
    
    Vector<double> diff_per_cell(triangulation.n_active_cells());

    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm);

    const double L2_error = diff_per_cell.l2_norm();
    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell, QGauss<dim>(quad_order+1),
                                      VectorTools::H1_seminorm);
    
    
    double H1_error = diff_per_cell.l2_norm();
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("L2 Error", L2_error);
    convergence_table.add_value("H1 Error", H1_error);
    
}

template <int dim>
void LaplaceProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<1; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':';
        if (cycle == 0)
        {
            /* Cube mesh */
//             GridGenerator::hyper_cube (triangulation, -1, 1);
            
            /* Read in mesh: */
            std::string mesh_name = "../../mesh/cylinder.ucd";
            read_in_mesh(mesh_name);
            // Set boundaries to neumann (boundary_id = 1)
//             typename Triangulation<dim>::cell_iterator
//             cell = triangulation.begin (),
//             endc = triangulation.end();
//             for (; cell!=endc; ++cell)
//             {
//                 for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//                 {
//                     if (cell->face(face)->at_boundary())
//                     {
//                         cell->face(face)->set_boundary_indicator (1);
//                     }
//                 }    
//             }
//            triangulation.refine_global (2);
        }
        else
        {
            triangulation.refine_global (1);
        }

        setup_system ();
        assemble_system ();
        solve ();
        process_solution (cycle);
        output_results_vtk(cycle);
        std::cout << " Done" << std::endl;
    }
    
    convergence_table.set_precision("L2 Error",8);
    convergence_table.set_scientific("L2 Error",true);
    
    convergence_table.set_precision("H1 Error",8);
    convergence_table.set_scientific("H1 Error",true);
    
    convergence_table.write_text(std::cout);
}
// END MAXWELL CLASS
int main (int argc, char* argv[])
{
    // read order as input via '-p x'
    unsigned int p_order=1;
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
    LaplaceProblem<3> maxwell(p_order);
    maxwell.run ();
    return 0;
}
