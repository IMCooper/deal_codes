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
    class Postprocessor; // DataPostprocessor
    void read_in_mesh (std::string mesh_file);
    void setup_system ();
    void assemble_system ();
    void solve ();
    void process_solution(const unsigned int cycle);
    void output_results_vtk (const unsigned int cycle) const;

    double calcErrorHcurlNorm();
//    double quicktest();
    //void mesh_info(const Triangulation<dim> &tria, const std::string        &filename);
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FE_Nedelec<dim>            fe;
//     FESystem<dim>          fe;
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
fe(order),
// fe (FE_Nedelec<dim>(order),1),//, FE_Nedelec<dim>(order), 1),
exact_solution()
{
    p_order = order;
    quad_order = p_order+2;
    
}
template <int dim>
MaxwellProblem<dim>::~MaxwellProblem ()
{
    dof_handler.clear ();
}

// Postprocessor for plotting:
template <int dim>
class MaxwellProblem<dim>::Postprocessor : public DataPostprocessor<dim>
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
MaxwellProblem<dim>::Postprocessor::Postprocessor ()
:
DataPostprocessor<dim>()
{}

template <int dim>
std::vector<std::string>
MaxwellProblem<dim>::Postprocessor::get_names() const
{
    std::vector<std::string> solution_names (dim, "E");
    solution_names.push_back ("curlE");
    solution_names.push_back ("curlE");
    solution_names.push_back ("curlE");
    
    return solution_names;
}
template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
MaxwellProblem<dim>::Postprocessor::
get_data_component_interpretation () const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    return interpretation;
}
template <int dim>
UpdateFlags
MaxwellProblem<dim>::Postprocessor::get_needed_update_flags() const
{
    return update_values | update_gradients | update_hessians | update_q_points; //update_normal_vectors |
}
template <int dim>
void
MaxwellProblem<dim>::Postprocessor::compute_derived_quantities_vector (const std::vector<Vector<double> >    &uh,
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

        // curlE:
        computed_quantities[q](0+dim) = (duh[q][2][1]-duh[q][1][2])*temp_scaling;
        computed_quantities[q](1+dim) = (duh[q][0][2]-duh[q][2][0])*temp_scaling;
        computed_quantities[q](2+dim) = (duh[q][1][0]-duh[q][0][1])*temp_scaling;
        
    }
}
// END Postprocessor class.

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
void MaxwellProblem<dim>::read_in_mesh (std::string mesh_name)
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
void MaxwellProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    DoFRenumbering::block_wise (dof_handler);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
       
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    // FE_Nedelec boundary condition.
    // Real part (begins at 0):
     VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 0, constraints);
//     VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 1, constraints);
//     VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 2, constraints);
    
    
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
       
    // Neumann storage
    std::vector<Vector<double> > neumann_value_list(n_face_q_points, Vector<double>(fe.n_components()));
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
        cell_matrix = 0;
        cell_rhs = 0;
        
        // Calc RHS values:
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_value_list);        

        // Loop over all element quad points:
//        std::cout << "New EL:" << std::endl;
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
//            std::cout << fe_values.quadrature_point(q_point) << " " << fe_values[E_re].value(4,q_point) << std::endl;
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
        // Loop over faces for neumann condition:
        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            fe_face_values.reinit (cell, face_number);
            if (cell->face(face_number)->at_boundary()
                &&
                (cell->face(face_number)->boundary_indicator() == 10))
            {
                // Store values of (mur^-1)*curl E:
                exact_solution.curl_value_list(fe_face_values.get_quadrature_points(), neumann_value_list);
                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    for (unsigned int component=0; component<dim; component++)
                    {
                        neumann_value_vector[component] = neumann_value_list[q_point](component);
                        normal_vector[component] = fe_face_values.normal_vector(q_point)(component);
                    }
                    cross_product(neumann_value, normal_vector, neumann_value_vector);
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        cell_rhs(i) += -(neumann_value*fe_face_values[E_re].value(i,q_point)*fe_face_values.JxW(q_point));                            

                    }
                }
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
template <int dim>
void MaxwellProblem<dim>::output_results_vtk (const unsigned int cycle) const
{
    
    std::ostringstream filename;
    filename << "solution-" << cycle << ".vtk";
    std::ofstream output (filename.str().c_str());
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    
    std::vector<std::string> solution_names (dim, "E_re");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    data_out.add_data_vector (solution, solution_names,
                              // new
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    

    /*
    Postprocessor postprocessor;
    data_out.add_data_vector(solution,postprocessor);
    */
    data_out.build_patches (10);
    data_out.write_vtk (output);
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
    typename Triangulation<dim>::cell_iterator cell,endc; // For querying cells.
    
    for (unsigned int cycle=0; cycle<3; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':';
        if (cycle == 0)
        {
            /* Cube mesh */
//            GridGenerator::hyper_cube (triangulation, -1, 1);
            /* move one vertex out of position */
//            cell = triangulation.begin_active(),
//            endc = triangulation.end();
//            for (; cell!=endc; ++cell)
//            {
//                Point<dim> &v1 = cell->vertex(7);
//                v1 +=Point<dim> (0.1,0,0);
//                Point<dim> &v2 = cell->vertex(3);
//                v2 +=Point<dim> (-0.1,0.2,0.15);
//            }

            /* subdivided Cube mesh */
//             GridGenerator::subdivided_hyper_cube (triangulation,2,-1.0,1.0);

            /* cube w/ slit mesh */
//             GridGenerator::hyper_cube_slit(triangulation);

    /* cuboid mesh */
//             GridGenerator::hyper_rectangle (triangulation, Point<dim>(-3.0,-2.0,-1.0), Point<dim>(1.0,2.0,3.0));

            
            
            /* parallelopiped mesh */
             Point<dim> corners[dim];
             corners[0]=Point<dim> (1.0,0.0,0.0);
             corners[1]=Point<dim> (0.5,1.0,0.0);
             corners[2]=Point<dim> (0.0,0.0,1.0);
             GridGenerator::parallelepiped(triangulation,corners);

            
            /* Cylinder mesh */
//             GridGenerator::cylinder(triangulation, 1.0, 1.0);
//             static const CylinderBoundary<dim> cylinder (1.0);
//             triangulation.set_boundary (0, cylinder);
            
            /* Sphere mesh */
//             GridGenerator::hyper_ball(triangulation,Point<dim>(0.0,0.0,0.0),1.0);
//             static const HyperBallBoundary<dim> sphere (Point<dim>(0.0,0.0,0.0),1.0);
//             triangulation.set_boundary(0,sphere);

    
           /* L shape mesh */
//             GridGenerator::hyper_L(triangulation);
            
            
            
            /* Read in mesh: */
//             std::string mesh_name = "../mesh/cylinder.ucd";
//             read_in_mesh(mesh_name);

            /* Set boundaries to neumann (boundary_id = 1) */
//             typename Triangulation<dim>::cell_iterator
//             cell = triangulation.begin (),
//             endc = triangulation.end();
//             for (; cell!=endc; ++cell)
//             {
//               //set bd flag.
//               for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//               {
//                   if (cell->face(face)->at_boundary())
//                   {
//                       cell->face(face)->set_all_boundary_indicators (1);
//                   }
//               }
//           }

            /* print cell boundary indicators: */
//             cell = triangulation.begin ();
//             endc = triangulation.end();
//             for (; cell!=endc; ++cell)
//             {
                //set bd flag.
//                 for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//                 {
//                     if (cell->face(face)->at_boundary())
//                     {
//                       std::cout << cell << " " << face << " " << (unsigned int)cell->face(face)->boundary_indicator() << std::endl;
//                     }       
//                 }
//             }


            /* refine mesh to start */
//           triangulation.refine_global(2);
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
