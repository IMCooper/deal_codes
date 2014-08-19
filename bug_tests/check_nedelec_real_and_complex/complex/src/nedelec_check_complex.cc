/*
 Solves the complex valued vector-wave equation:
 curl(curl(E)) + kappa*E = Js
 
 kappa=1+i
 
 Js = kappa*E.
 
 which has soluion:
 E = (x^2 + i x^2, y^2 + i y^2, z^2 + i z^2).
 
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
Function<dim> (dim+dim)
{}
// Wave Propagation members:

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
        value_list[i](0) = 0.0;//-p(0)*p(0);
        value_list[i](1) = 0.0;//-p(1)*p(1);
        value_list[i](2) = 0.0;//-p(2)*p(2);
        // Imaginary:
        value_list[i](3) = 2.0*p(0)*p(0);
        value_list[i](4) = 2.0*p(1)*p(1);
        value_list[i](5) = 2.0*p(2)*p(2);
    }
}
// END RHS

// Dirichlet BCs:.
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
Function<dim> (dim+dim)
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
        // Imaginary:
        value_list[i](3) = p(0)*p(0);
        value_list[i](4) = p(1)*p(1);
        value_list[i](5) = p(2)*p(2);
    }
}
// Additional functions to create Neumann conditions.
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
        // Imaginary:
        value_list[i](3) = 0.0;
        value_list[i](4) = 0.0;
        value_list[i](5) = 0.0;
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
    double dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const;
    double dotprod(const Tensor<1,dim> &A, const Vector<double> &B) const;
    class ComputeH; // DataPostprocessor
    //void read_in_mesh (std::string mesh_file);
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
    //FE_Nedelec<dim>            fe;
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
    
    Vector<double> param_mur;
    Vector<double> param_kappa_re;
    Vector<double> param_kappa_im;
};

template <int dim>
MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
:
dof_handler (triangulation),
// Defined as FESystem, and we need 2 FE_Nedelec - first (0) is real part, second (1) is imaginary part.
fe (FE_Nedelec<dim>(order), 2),//, FE_Nedelec<dim>(order), 1),
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

template<int dim>
double MaxwellProblem<dim>::dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A[k]*B[k];
    }
    return return_val;
}

template<int dim>
double MaxwellProblem<dim>::dotprod(const Tensor<1,dim> &A, const Vector<double> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A[k]*B(k);
    }
    return return_val;
}

// Compute magnetic field class:
// i.e. Want to compute H = i(omega/mu)*curlE & compare to actual soln.
//Naive way:
template <int dim>
class MaxwellProblem<dim>::ComputeH : public DataPostprocessor<dim>
{
public:
    ComputeH ();
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
MaxwellProblem<dim>::ComputeH::ComputeH ()
:
DataPostprocessor<dim>()
{}

template <int dim>
std::vector<std::string>
MaxwellProblem<dim>::ComputeH::get_names() const
{
    std::vector<std::string> solution_names (dim, "E_re");
    solution_names.push_back ("E_im");
    solution_names.push_back ("E_im");
    solution_names.push_back ("E_im");
    solution_names.push_back ("curlE_re");
    solution_names.push_back ("curlE_re");
    solution_names.push_back ("curlE_re");
    solution_names.push_back ("curlE_im");
    solution_names.push_back ("curlE_im");
    solution_names.push_back ("curlE_im");
    
    return solution_names;
}
template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
MaxwellProblem<dim>::ComputeH::
get_data_component_interpretation () const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    // for H re/imag:
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    return interpretation;
}
template <int dim>
UpdateFlags
MaxwellProblem<dim>::ComputeH::get_needed_update_flags() const
{
    return update_values | update_gradients | update_hessians | update_q_points; //update_normal_vectors |
}
template <int dim>
void
MaxwellProblem<dim>::ComputeH::compute_derived_quantities_vector (const std::vector<Vector<double> >    &uh,
                                                  const std::vector<std::vector<Tensor<1,dim> > >      &duh,
                                                  const std::vector<std::vector<Tensor<2,dim> > >      &dduh,
                                                  const std::vector<Point<dim> >                       &normals,
                                                  const std::vector<Point<dim> >                       &evaluation_points,
                                                  std::vector<Vector<double> >                         &computed_quantities) const
{
    const unsigned int n_quadrature_points = uh.size();
    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
    Assert (uh[0].size() == dim+dim,                            ExcInternalError());
    
    double temp_scaling = 1.0;
    
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        // Electric field, E:
        for (unsigned int d=0; d<dim; ++d)
        {
            computed_quantities[q](d) = uh[q](d);
            computed_quantities[q](d+dim) = uh[q](d+dim);
        }

        // Magnetic field, H = i*omega*curl(E).
        // i.e. flip the real/imag parts of curl(E) and multiply imaginary part by -1.
        // Note that we need to scale by mu0 too (i.e. temp_scaling factors in 1/mu_0)
        //real part:
        computed_quantities[q](0+2*dim) = -(duh[q][5][1]-duh[q][4][2])*temp_scaling;
        computed_quantities[q](1+2*dim) = -(duh[q][3][2]-duh[q][5][0])*temp_scaling;
        computed_quantities[q](2+2*dim) = -(duh[q][4][0]-duh[q][3][1])*temp_scaling;
        //imaginary part
        computed_quantities[q](0+3*dim) = (duh[q][2][1]-duh[q][1][2])*temp_scaling;
        computed_quantities[q](1+3*dim) = (duh[q][0][2]-duh[q][2][0])*temp_scaling;
        computed_quantities[q](2+3*dim) = (duh[q][1][0]-duh[q][0][1])*temp_scaling;
        
    }
}
// END Compute magnetic field class

template<int dim>
double MaxwellProblem<dim>::calcErrorHcurlNorm()
{
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
     
     FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values    |  update_gradients |
                              update_quadrature_points  |  update_JxW_values);
     
     // Extractors to real and imaginary parts
     const FEValuesExtractors::Vector E_re(0);
     const FEValuesExtractors::Vector E_im(dim);
     
     const unsigned int dofs_per_cell = fe.dofs_per_cell;
     
     std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
     
     // storage for exact sol:
     std::vector<Vector<double> > exactsol(n_q_points, Vector<double>(fe.n_components()));
     std::vector<Vector<double> > exactcurlsol(n_q_points, Vector<double>(fe.n_components()));
     Tensor<1,dim>  exactsol_re;
     Tensor<1,dim>  exactsol_im;
     Tensor<1,dim>  exactcurlsol_re;
     Tensor<1,dim>  exactcurlsol_im;

     // storage for computed sol:
     std::vector<Tensor<1,dim> > sol_re(n_q_points);
     std::vector<Tensor<1,dim> > sol_im(n_q_points);
     Tensor<1,dim> curlsol_re(n_q_points);
     Tensor<1,dim> curlsol_im(n_q_points);
     
     double h_curl_norm=0.0;
     
     unsigned int block_index_i;
     
     typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler.begin_active(),
     endc = dof_handler.end();
     for (; cell!=endc; ++cell)
     {
         fe_values.reinit (cell);
         
         // Store exact values of E and curlE:
         exact_solution.vector_value_list(fe_values.get_quadrature_points(), exactsol);
         exact_solution.curl_value_list(fe_values.get_quadrature_points(), exactcurlsol);


         // Store computed values at quad points:
         fe_values[E_re].get_function_values(solution, sol_re);
         fe_values[E_im].get_function_values(solution, sol_im);
         
         // Calc values of curlE from fe solution:
         cell->get_dof_indices (local_dof_indices);
         // Loop over quad points to calculate solution:
         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {
             // Split exact solution into real/imaginary parts:
             for (unsigned int component=0;component<dim;component++)
             {
                 exactsol_re[component] = exactsol[q_point][component];
                 exactsol_im[component] = exactsol[q_point][component+dim];
                 exactcurlsol_re[component] = exactcurlsol[q_point][component];
                 exactcurlsol_im[component] = exactcurlsol[q_point][component+dim];
             }
             // Loop over DoFs to calculate curl of solution @ quad point
             curlsol_re=0.0;
             curlsol_im=0.0;
             for (unsigned int i=0; i<dofs_per_cell; ++i)
             {
                 block_index_i = fe.system_to_block_index(i).first;
                 // Construct local curl value @ quad point
                 if (block_index_i==0)
                 {
                     curlsol_re += solution(local_dof_indices[i])*fe_values[E_re].curl(i,q_point);
                 }
                 else if (block_index_i==1)
                 {
                     curlsol_im += solution(local_dof_indices[i])*fe_values[E_im].curl(i,q_point);
                 }
             }
             // Integrate difference at each point:             
             h_curl_norm += ( (exactsol_re-sol_re[q_point])*(exactsol_re-sol_re[q_point])
                             + (exactsol_im-sol_im[q_point])*(exactsol_im-sol_im[q_point])
                             + (exactcurlsol_re-curlsol_re)*(exactcurlsol_re-curlsol_re)
                             + (exactcurlsol_im-curlsol_im)*(exactcurlsol_im-curlsol_im)
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
    // Real part (begins at 0):
     VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 0, constraints);
    // Imaginary part (begins at dim):
     VectorTools::project_boundary_values_curl_conforming(dof_handler, dim, exact_solution, 0, constraints);
    
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close ();
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,false);
    
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
    
    // Setup material parameters:
    param_mur.reinit(1);
    param_mur(0) = 1.0;
    
    //  kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    param_kappa_re.reinit(1);
    param_kappa_re(0) = 1.0;
    param_kappa_im.reinit(1);
    param_kappa_im(0) = 1.0;
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
        
    // Extractors to real and imaginary parts
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
   
    // Local cell storage:
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    // block indices:
    unsigned int block_index_i;
    unsigned int block_index_j;
    
    //RHS storage:
    RightHandSide<dim> right_hand_side;
    std::vector<Vector<double> > rhs_value_list(n_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim> rhs_value_list_re(dim);
    Tensor<1,dim> rhs_value_list_im(dim);
    
       
    // Neumann storage
    std::vector<Vector<double> > neumann_value_list(n_face_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim> neumann_value_list_re(dim);
    Tensor<1,dim> neumann_value_list_im(dim);
    Tensor<1,dim> neumann_value_re(dim);
    Tensor<1,dim> neumann_value_im(dim);
    Tensor<1,dim> normal_vector;
    
    // Material parameters:
    double current_mur;
    double current_kappa_re;
    double current_kappa_im;
    
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        current_mur = param_mur(cell->material_id());
        current_kappa_re = param_kappa_re(cell->material_id());
        current_kappa_im = param_kappa_im(cell->material_id());
        cell_matrix = 0;
        cell_rhs = 0;
        
        // Calc RHS values:
        right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_value_list);        

        // Loop over all element quad points:
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            // separate real/imag components of rhs value:
            for (unsigned int component=0; component<dim; component++)
            {
                rhs_value_list_re[component] = rhs_value_list[q_point](component);
                rhs_value_list_im[component] = rhs_value_list[q_point](component+dim);
            }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                block_index_i = fe.system_to_block_index(i).first;
                // Construct local matrix:
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    block_index_j = fe.system_to_block_index(j).first;
                    // block 0 = real, block 1 = imaginary.
                    if (block_index_i == block_index_j)
                    {
                        if (block_index_i == 0)
                        {
                            cell_matrix(i,j) += ( (1.0/current_mur)*(fe_values[E_re].curl(i,q_point)*fe_values[E_re].curl(j,q_point))
                                                 + current_kappa_re*fe_values[E_re].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                                 )*fe_values.JxW(q_point);
                        }
                        else if (block_index_i == 1)
                        {
                            cell_matrix(i,j) += ( (1.0/current_mur)*(fe_values[E_im].curl(i,q_point)*fe_values[E_im].curl(j,q_point))
                                                 + current_kappa_re*fe_values[E_im].value(i,q_point)*fe_values[E_im].value(j,q_point)
                                                 )*fe_values.JxW(q_point);
                        }
                    }
                    else
                    {
                        if (block_index_i == 0) // then block_index_j == 1
                        {                            
                            cell_matrix(i,j) += -current_kappa_im*(   fe_values[E_re].value(i,q_point)*fe_values[E_im].value(j,q_point)
                                                                  )*fe_values.JxW(q_point);
                        }
                        else if (block_index_i == 1) // then block_index_j == 0
                        {                            
                            cell_matrix(i,j) += current_kappa_im*(  fe_values[E_im].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                                                  )*fe_values.JxW(q_point);
                        }
                    }
                }
                // construct local RHS:
                if (block_index_i == 0)
                {
                    cell_rhs(i) += rhs_value_list_re*fe_values[E_re].value(i,q_point)*fe_values.JxW(q_point);
                }
                else if (block_index_i == 1)
                {
                    cell_rhs(i) += rhs_value_list_im*fe_values[E_im].value(i,q_point)*fe_values.JxW(q_point);
                }                
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
//                         neumann_value_list_re[component] = neumann_value_list[q_point](component);
//                         neumann_value_list_im[component] = neumann_value_list[q_point](component+dim);
//                         normal_vector[component] = fe_face_values.normal_vector(q_point)(component);
//                     }
//                     cross_product(neumann_value_re, normal_vector, neumann_value_list_re);
//                     cross_product(neumann_value_im, normal_vector, neumann_value_list_im);
//                     for (unsigned int i=0; i<dofs_per_cell; ++i)
//                     {
//                         block_index_i = fe.system_to_block_index(i).first;
//                         if (block_index_i == 0) // then block_index_j == 1
//                         {
//                             cell_rhs(i) += -(neumann_value_re*fe_face_values[E_re].value(i,q_point)*fe_face_values.JxW(q_point));                            
//                         }
//                         else
//                         {
//                             cell_rhs(i) += -(neumann_value_im*fe_face_values[E_im].value(i,q_point)*fe_face_values.JxW(q_point));
//                         }
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
    /* old way:
    std::vector<std::string> solution_names;
    switch (dim)
    {
        case 1:
            solution_names.push_back ("Er");
            solution_names.push_back ("Ei");
            break;
        case 2:
            solution_names.push_back ("Er1");
            solution_names.push_back ("Er2");
            solution_names.push_back ("Ei1");
            solution_names.push_back ("Ei2");
            break;
        case 3:
            solution_names.push_back ("Er1");
            solution_names.push_back ("Er2");
            solution_names.push_back ("Er3");
            solution_names.push_back ("Ei1");
            solution_names.push_back ("Ei2");
            solution_names.push_back ("Ei3");
            break;
        default:
            Assert (false, ExcNotImplemented());
    }
    */
    // new way:
    std::vector<std::string> solution_names (dim, "E_re");
    solution_names.push_back ("E_im");
    solution_names.push_back ("E_im");
    solution_names.push_back ("E_im");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
    
    data_out.add_data_vector (solution, solution_names,
                              // new
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches (p_order+2);
    data_out.write_vtk (output);
}
template<int dim>
void MaxwellProblem<dim>::process_solution(const unsigned int cycle)
{
    // Masks for real & imaginary parts
    const ComponentSelectFunction<dim> E_re_mask (std::make_pair(0,dim), dim+dim);
    const ComponentSelectFunction<dim> E_im_mask (std::make_pair(dim, dim+dim), dim+dim);
    
    Vector<double> diff_per_cell_re(triangulation.n_active_cells());
    Vector<double> diff_per_cell_im(triangulation.n_active_cells());
    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell_re, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm,
                                      &E_re_mask);
    
    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell_im, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm,
                                      &E_im_mask);
    
    const double L2_error_re = diff_per_cell_re.l2_norm();
    const double L2_error_im = diff_per_cell_im.l2_norm();
    const double L2_error = sqrt(L2_error_re*L2_error_re + L2_error_im*L2_error_im);
    
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
    for (unsigned int cycle=0; cycle<3; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':';
        if (cycle == 0)
        {
            /* Cube mesh */
            GridGenerator::hyper_cube (triangulation, -1, 1);
//             Set boundaries to neumann (boundary_id = 1)
//              typename Triangulation<dim>::cell_iterator
//              cell = triangulation.begin (),
//              endc = triangulation.end();
//              for (; cell!=endc; ++cell)
//              {
//                  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//                  {
//                      if (cell->face(face)->at_boundary())
//                      {
//                          cell->face(face)->set_boundary_indicator (1);
//                      }
//                  }          
//              }
           triangulation.refine_global (1);
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
