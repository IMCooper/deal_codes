#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

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
#include <deal.II/numerics/data_postprocessor.h>

//#include <boost/math/special_functions/bessel.hpp> // Req'd for bessel functions in L-shape solution w/ singularity.

#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <locale>
#include <string>

namespace Maxwell
{
    using namespace dealii;
    const double constant_PI = numbers::PI;
    
    namespace IO_Data
    {
        std::string output_filename = "solution";
        std::string output_filetype = "vtk";
    }
    
    //    template <int dim>
    namespace EquationData
    {
        double constant_epsilon0 = 0.0; // electric constant (permittivity)
        double constant_mu0 = 1.25663706e-6; // magnetic constant (permeability)
        double constant_sigma0 = 0.01; // background conductivity, set to regularise system (can't solve curlcurlE = f).
        // Material Parameters:
        double param_omega=1.0; // angular frequency, rads/sec.
        double param_epsilon_star=1.0e-6;
        double param_sigma_star=0.0;
        double param_mu_star=1.0;
        
        
        /* Vectors holding equation parameters
         * intially assume 2 different objects.
         * This can be adjusted via the parameter file and then
         * use vector.reinit(number_of objects) later on.
         */
        Vector<double> param_mur;
        Vector<double> param_sigma;
        Vector<double> param_epsilon;
        Vector<double> param_kappa_re; // Kappa = Kappa_re + i*Kappa_im = -omega.^2*epr + i*omega*sigma
        Vector<double> param_kappa_im;
        
        Point<3> source_point(1.1,1.1,1.1);
        Tensor<1,3> coil_direction;
    }
    
    /* ExactSolution class:
     * contains solution and functions for use with Dirichlet or Neumann conditions.
     */
    template<int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution(); //const FullMatrix<double> polTensor_in_re(dim), const FullMatrix<double> &polTensor_in_im);
        
        virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                        std::vector<Vector<double> >   &values) const;

    private:
        double dotprod(const Vector<double> &A, const Vector<double> &B) const;
        double dotprod(const Vector<double> &A, const Point<dim> &B) const;
    };
    
    template<int dim>
    ExactSolution<dim>::ExactSolution()
    :
    Function<dim> (dim+dim)
    {}
    
    // ExactSolution members:
    template<int dim>
    double ExactSolution<dim>::dotprod(const Vector<double> &A, const Vector<double> &B) const
    {
        double return_val = 0;
        for(unsigned int k = 0; k < dim; k++) {
            return_val += A(k)*B(k);
        }
        return return_val;
    }
    template<int dim>
    double ExactSolution<dim>::dotprod(const Vector<double> &A, const Point<dim> &B) const
    {
        double return_val = 0;
        for(unsigned int k = 0; k < dim; k++) {
            return_val += A(k)*B(k);
        }
        return return_val;
    }
    
    template <int dim>
    void ExactSolution<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> > &value_list) const
    {
        Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
        const unsigned int n_points = points.size();
        
        Tensor<1,dim> shifted_point;
        Tensor<1,dim> result;
        
        for (unsigned int k=0; k<points.size(); ++k)
        {
            const Point<dim> &p = points[k];
            /* Work out the vector (stored as a tensor so we can use cross_product)
             * from the source point to the current point, p
             */
            for (unsigned int i =0;i<dim;i++)
            {
                shifted_point[i] = p(i) - EquationData::source_point(i);
            }
            double rad = p.distance(EquationData::source_point);
            double factor = 1.0/(4.0*constant_PI*rad*rad*rad);
            
            cross_product(result, EquationData::coil_direction, shifted_point);
            result *= factor;
            for (unsigned int i=0;i<dim;i++)
            {
                // Real
                value_list[k](i) = result[i];
                // Imaginary
                value_list[k](i+dim) = 0.0; // minus as the solution is the complex conjugate of what we need.
            }
        }
    }
    // END EXACTSOLUTION CLASS
    
    // MAIN MAXWELL CLASS
    template <int dim>
    class MaxwellProblem
    {
    public:
        MaxwellProblem (const unsigned int order);
        ~MaxwellProblem ();
        void run ();
        
        Point<dim> centre_of_sphere;
        double sphere_radius;
       
    private:
        double dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const;
        double dotprod(const Tensor<1,dim> &A, const Vector<double> &B) const;
        
        void read_in_mesh (std::string mesh_name);
        void setup_system ();
        void assemble_system ();
        void solve ();
        void process_solution(const unsigned int cycle);
        void output_results_eps (const unsigned int cycle) const;
        void output_results_vtk (const unsigned int cycle) const;
        void output_results_gmv(const unsigned int cycle) const;
        class Postprocessor; // If setting up class within this class for DataPostprocessor.
        //void mesh_info(const Triangulation<dim> &tria, const std::string        &filename);
        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dof_handler;
        FESystem<dim>          fe;
        ConstraintMatrix     constraints;
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double>       solution;
        Vector<double>       system_rhs;
        
        ConvergenceTable	   convergence_table;
        
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
    // Defined as FESystem, and we need 2 FE_Nedelec - first (0) is real part, second (1) is imaginary part.
    // Then need to use system blocks to solve for them.
    fe (FE_Nedelec<dim>(order), 1, FE_Nedelec<dim>(order), 1)
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
        
        // Setup material parameters (could be expanded into own routine later):
        // e.g. Could be based on number of material_id's read in.
        /* Note:
         * Have multiplied equation through by mu_0.
         * i.e. mu is now mur (smallest value is 1)
         *      and kappa must be multiplied by mu_0 in both the real and imaginary parts.
         */
        EquationData::param_mur.reinit(1);
        EquationData::param_mur(0) = EquationData::param_mu_star;
        
        EquationData::param_sigma.reinit(1);
        EquationData::param_sigma(0) = EquationData::param_sigma_star;
        
        EquationData::param_epsilon.reinit(1);
        EquationData::param_epsilon(0) = EquationData::param_epsilon_star;
        
        
        // kappa = -omega^2*epr + i*omega*sigma;
        // i.e. kappa_re = -omega^2
        //      kappa_im = omega*sigma
        EquationData::param_kappa_re.reinit(EquationData::param_mur.size());
        EquationData::param_kappa_im.reinit(EquationData::param_mur.size());
        for (unsigned int i=0;i<EquationData::param_mur.size();i++) // note mur and kappa must have same size:
        {
            EquationData::param_kappa_re(i) = 1e-6;//-EquationData::param_omega*EquationData::param_omega*EquationData::param_epsilon(i);
            EquationData::param_kappa_im(i) = 0.0;//EquationData::param_omega*EquationData::param_sigma(i);
        }
        EquationData::coil_direction[0]=1.0;
        EquationData::coil_direction[1]=1.0;
        EquationData::coil_direction[2]=1.0;
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
            current_mur = EquationData::param_mur(0);
            current_kappa_re = EquationData::param_kappa_re(0);
            current_kappa_im = EquationData::param_kappa_im(0);
            cell_matrix = 0;
            cell_rhs = 0;
            
            // Loop over quad points:
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
                
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
                    
                    // RHS (J_S) is zero so nothing to do for cell_rhs as it is already zero.
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
                    // For this problem, vector value list returns values of H
                    // Note that H = i(omega/mu)*curl(E), (mu NOT mur, remember mur = mu/mu0)
                    // so (1/mur)*curl(E) = mu_0*H/(i*omega). (1/i = -i)
                    // i.e. we use the imaginary part of H for real curl E and real for imag curl E.
                    //      and must multiply the imag part by -1.
                    exact_solution.vector_value_list(fe_face_values.get_quadrature_points(), neumann_value_list);
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        for (unsigned int component=0; component<dim; component++)
                        {
                            neumann_value_list_re[component] = neumann_value_list[q_point](component+dim);
                            neumann_value_list_im[component] = -neumann_value_list[q_point](component);
                            normal_vector[component] = fe_face_values.normal_vector(q_point)(component);
                        }
                        cross_product(neumann_value_re, normal_vector, neumann_value_list_re);
                        cross_product(neumann_value_im, normal_vector, neumann_value_list_im);
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            block_index_i = fe.system_to_block_index(i).first;
                            if (block_index_i == 0) // then block_index_j == 1
                            {
                                cell_rhs(i) += -(neumann_value_re*fe_face_values[E_re].value(i,q_point)*fe_face_values.JxW(q_point));
                            }
                            else
                            {
                                cell_rhs(i) += -(neumann_value_im*fe_face_values[E_im].value(i,q_point)*fe_face_values.JxW(q_point));
                            }
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
        
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());
        convergence_table.add_value("L2 Error", L2_error);
    }
    
    // Compute magnetic field class:
    // i.e. Want to compute H = i(omega/mu)*curlE & compare to actual soln.
    //Naive way:
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
//    private:
        ExactSolution<dim> exact_solution;
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
        std::vector<std::string> solution_names (dim, "E_re");
        solution_names.push_back ("E_im");
        solution_names.push_back ("E_im");
        solution_names.push_back ("E_im");
         solution_names.push_back ("H_re");
         solution_names.push_back ("H_re");
         solution_names.push_back ("H_re");
         solution_names.push_back ("H_im");
         solution_names.push_back ("H_im");
         solution_names.push_back ("H_im");
         
         solution_names.push_back ("Error_re");
         solution_names.push_back ("Error_re");
         solution_names.push_back ("Error_re");
         solution_names.push_back ("Error_im");
         solution_names.push_back ("Error_im");
         solution_names.push_back ("Error_im");
        
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
        // for curlE re/imag:
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
         
         // For perturbed field error:
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
        Assert (uh[0].size() == dim+dim,                            ExcInternalError());
        
        std::vector<Vector<double> > solution_value_list(n_quadrature_points, Vector<double>(dim+dim));
        exact_solution.vector_value_list(evaluation_points, solution_value_list);
        
        for (unsigned int q=0; q<n_quadrature_points; ++q)
        {
            // Electric field, E:
            for (unsigned int d=0; d<dim+dim; ++d)
            {
                computed_quantities[q](d) = uh[q](d);
            }
            
            // CurlE:
            //real part:
            computed_quantities[q](0+2*dim) = (duh[q][2][1]-duh[q][1][2]);
            computed_quantities[q](1+2*dim) = (duh[q][0][2]-duh[q][2][0]);
            computed_quantities[q](2+2*dim) = (duh[q][1][0]-duh[q][0][1]);
            //imaginary part
            computed_quantities[q](0+3*dim) = (duh[q][5][1]-duh[q][4][2]);
            computed_quantities[q](1+3*dim) = (duh[q][3][2]-duh[q][5][0]);
            computed_quantities[q](2+3*dim) = (duh[q][4][0]-duh[q][3][1]);
            
            // Error: E - E_known:
            for (unsigned int i=0;i<dim+dim;i++)
            {
                computed_quantities[q](i+4*dim) = uh[q](i)-solution_value_list[q](i);
            }
        }
    }
    
    
    template <int dim>
    void MaxwellProblem<dim>::output_results_vtk (const unsigned int cycle) const
    {
        
        std::ostringstream filename;
        filename << IO_Data::output_filename << "-" << cycle << "." << IO_Data::output_filetype;
        std::ofstream output (filename.str().c_str());
        
        // postprocessor handles all quantities to output
        // NOTE IT MUST GO BEFORE DataOut<dim>!!!
        Postprocessor postprocessor;
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        
        data_out.add_data_vector(solution, postprocessor);
        data_out.build_patches (quad_order);
        data_out.write_vtk (output);
    }
    
    template <int dim>
    void MaxwellProblem<dim>::run ()
    {
        
        for (unsigned int cycle=0; cycle<3; ++cycle)
        {
            std::cout << "Cycle " << cycle << ':' << std::endl;
            if (cycle == 0)
            {
                // Cube mesh
                            GridGenerator::hyper_cube (triangulation, -1, 1);
                            triangulation.refine_global (1);
                
                
                // Set boundaries to neumann (boundary_id = 1)
//                 typename Triangulation<dim>::cell_iterator
//                 cell = triangulation.begin (),
//                 endc = triangulation.end();
//                 for (; cell!=endc; ++cell)
//                 {
//                     for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//                     {
//                         if (cell->face(face)->at_boundary())
//                         {
//                             cell->face(face)->set_boundary_indicator (10);
//                         }
//                     }
//                 }
                //             triangulation.refine_global (1);
            }
            else
                triangulation.refine_global (1);
            std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
            setup_system ();
            std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
            assemble_system ();
            solve ();
            process_solution (cycle);
            output_results_vtk (cycle);
        }
        convergence_table.set_precision("L2 Error",8);
        convergence_table.set_scientific("L2 Error",true);
        
        convergence_table.write_text(std::cout);
    }
    // END MAXWELL CLASS
}


int main (int argc, char* argv[])
{
    using namespace Maxwell;
    
    
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
