
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_bessel.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
//#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
namespace Maxwell_FEM
{
    using namespace dealii;
    
    // EXACT SOLUTION CLASS
    template<int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>() {}
        virtual double value (const Point<dim> &p,
                              const unsigned int component) const;
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double> &result) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double> &values,
                                 const unsigned int component) const;
        virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                        std::vector<Vector<double> >   &values) const;
        virtual Tensor<1,dim> gradient (const Point<dim> &p,
                                        const unsigned int component) const;
        virtual void gradient_list (const std::vector<Point<dim> > &points,
                                    std::vector< Tensor< 1, dim > > &gradients,
                                    const unsigned int component) const;
        virtual void vector_gradient_list (const std::vector<Point<dim> > &points,
                                           std::vector<std::vector<Tensor<1,dim> > > &gradients) const;
    private:
        const double PI = dealii::numbers::PI;
    };
    // DEFINE EXACT SOLUTION MEMBERS
    template<int dim>
    double ExactSolution<dim>::value(const Point<dim> &p,
                                     const unsigned int component) const
    {
        Assert (dim >= 2, ExcNotImplemented());
        AssertIndexRange(component, dim);
        
        //2D solution
        double val = -1000;
        switch(component) {
            case 0:	val = cos(PI*p(0))*sin(PI*p(1));
            case 1:	val = -sin(PI*p(0))*cos(PI*p(1));
        }
        return val;
        
    }
    template<int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &result) const
    {
        Assert(dim >= 2, ExcNotImplemented());
        result(0) = cos(PI*p(0))*sin(PI*p(1));
        result(1) = -sin(PI*p(0))*cos(PI*p(1));
        
    }
    template <int dim>
    void ExactSolution<dim>::value_list (const std::vector<Point<dim> > &points,
                                         std::vector<double> &values,
                                         const unsigned int component) const
    {
        Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
        AssertIndexRange(component, dim);
        for (unsigned int i=0; i<points.size(); ++i)
        {
            const Point<dim> &p = points[i];
            switch(component)
            {
                case 0:
                    values[i] = cos(PI*p(0))*sin(PI*p(1));
                case 1:
                    values[i] = -sin(PI*p(0))*cos(PI*p(1));
            }
        }
    }
    template <int dim>
    void ExactSolution<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> >   &values) const
    {
        Assert (dim >= 2, ExcNotImplemented());
        Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
        
        for (unsigned int i=0; i<points.size(); ++i)
        {
            const Point<dim> &p = points[i];
            values[i](0) = cos(PI*p(0))*sin(PI*p(1));
            values[i](1) = -sin(PI*p(0))*cos(PI*p(1));

        }
        
    }
    
    template <int dim>
    Tensor<1,dim> ExactSolution<dim>::gradient (const Point<dim> &p,
                                               const unsigned int component) const
    {
        Assert (dim >= 2, ExcNotImplemented());
        AssertIndexRange(component, dim);
        
        Tensor<1,dim> result;
        switch(component) {
            case 0:
                result[0] = -PI*sin(PI*p(0))*sin(PI*p(1));
                result[1] = PI*cos(PI*p(0))*cos(PI*p(1));
            case 1:
                result[0] = -PI*cos(PI*p(0))*cos(PI*p(1));
                result[1] = PI*sin(PI*p(0))*sin(PI*p(1));
        }
        return result;
    }
    template <int dim>
    void ExactSolution<dim>::gradient_list (const std::vector<Point<dim> > &points,
                                            std::vector< Tensor< 1, dim > > &gradients,
                                            const unsigned int component) const
    {
        Assert (gradients.size() == points.size(), ExcDimensionMismatch(gradients.size(), points.size()));
        AssertIndexRange(component, dim);
        for (unsigned int i=0; i<points.size(); ++i)
        {
            const Point<dim> &p = points[i];
            Tensor<1,dim> &result = gradients[i];
            switch(component)
            {
                case 0:
                    result[0] = -PI*sin(PI*p(0))*sin(PI*p(1));
                    result[1] = PI*cos(PI*p(0))*cos(PI*p(1));
                case 1:
                    result[0] = -PI*cos(PI*p(0))*cos(PI*p(1));
                    result[1] = PI*sin(PI*p(0))*sin(PI*p(1));
            }
        }
        
    }
    template <int dim>
    void ExactSolution<dim>::vector_gradient_list (const std::vector<Point<dim> > &points,
                                                   std::vector<std::vector<Tensor<1,dim> > > &gradients) const
    {
        AssertVectorVectorDimension(gradients, points.size(), dim);
        for (unsigned int i=0; i<points.size(); ++i)
        {
            const Point<dim> &p = points[i];
            gradients[i][0][0] = -PI*sin(PI*p(0))*sin(PI*p(1));
            gradients[i][0][1] = PI*cos(PI*p(0))*cos(PI*p(1));
            //
            gradients[i][1][0] = -PI*cos(PI*p(0))*cos(PI*p(1));
            gradients[i][1][1] = PI*sin(PI*p(0))*sin(PI*p(1));
        }
    }
    // END EXACT SOLUTION MEMBERS
    
    // RIGHT HAND SIDE CLASS
    template <int dim>
    class RightHandSide :  public Function<dim>
    {
    public:
        RightHandSide ();
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const;
        virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                        std::vector<Vector<double> >   &value_list) const;
    private:
        const double PI = dealii::numbers::PI;
    };
    // DEFINE RIGHT HAND SIDE MEMBERS
    template <int dim>
    RightHandSide<dim>::RightHandSide () :
    Function<dim> (dim)
    {}
    template <int dim>
    inline
    void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
    {
        Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));
        Assert (dim >= 2, ExcNotImplemented());
        
        //2D solution
        values(0) = (2*PI*PI + 1)*cos(PI*p(0))*sin(PI*p(1));
        values(1) = -(2*PI*PI + 1)*sin(PI*p(0))*cos(PI*p(1));
    }
    template <int dim>
    void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> >   &value_list) const
    {
        Assert (value_list.size() == points.size(), ExcDimensionMismatch (value_list.size(), points.size()));
        const unsigned int n_points = points.size();
        for (unsigned int p=0; p<n_points; ++p)
        {
            RightHandSide<dim>::vector_value (points[p], value_list[p]);
        }
    }
    // END RIGHT HAND SIDE MEMBERS
    
    
    // BOUNDARY VALUE CLASS
    template <int dim>
    class DirichletBoundaryValues : public Function<dim>
    {
    public:
        DirichletBoundaryValues ();
        //        virtual double value(const Point<dim> &p) const;
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
        virtual void vector_value_list(const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const;
    private:
        const double PI = dealii::numbers::PI;
        
    };
    // DEFINE BOUNDARY VALUE MEMBERS
    template <int dim>
    DirichletBoundaryValues<dim>::DirichletBoundaryValues ()
    :
    Function<dim> (dim)
    {}
    
    //this can be used for 2D BCs, since we only have one component to deal with
    
    //template <int dim>
    //double DirichletBoundaryValues<dim>::value(const Point<dim> &/*p*/) const
    //{
    //    return 0;
    //
    //}
    
    template <int dim>
    inline
    void DirichletBoundaryValues<dim>::vector_value (const Point<dim> &/*p*/,
                                                     Vector<double> &values) const
    {
        Assert (values.size() == 2, ExcDimensionMismatch (values.size(), 2));
        values(0) = 1;
        values(1) = 1;
    }
    
    template <int dim>
    void DirichletBoundaryValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                          std::vector<Vector<double> > &value_list) const
    {
        Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
        const unsigned int n_points = points.size();
        for (unsigned int p=0; p<n_points; p++) {
            DirichletBoundaryValues<dim>::vector_value(points[p],value_list[p]);
        }
    }
    // END BOUNDARY VALUE MEMBERS
    
    // MAXWELL PROBLEM CLASS
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
        //    double dotprod(const Vector<double> &A, const Vector<double> &B) const;
        void setup_system ();
        void assemble_system ();
        unsigned int solve ();
        void refine_grid ();
        void process_solution(const unsigned int cycle);
        void output_results_eps (const unsigned int cycle) const;
        void output_results_vtk (const unsigned int cycle) const;
        void output_results_gmv(const unsigned int cycle) const;
        
        ConditionalOStream pcout;
        Triangulation<dim>   triangulation;
        //        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim>      dof_handler;
        FE_Nedelec<dim>      fe;
        //    FESystem<dim>        fe;
        ConstraintMatrix     hanging_node_constraints;
        PETScWrappers::MPI::SparseMatrix system_matrix;
        PETScWrappers::MPI::Vector       solution;
        PETScWrappers::MPI::Vector       system_rhs;
        MPI_Comm mpi_communicator;
        unsigned int p_order;
        unsigned int quad_order;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;
	
	std::vector<types::global_dof_index> local_dofs_per_process;
	types::global_dof_index n_local_dofs;
	unsigned int         n_local_cells;
        
        ConvergenceTable	   convergence_table;
    };
    // DEFINE MAXWELL PROBLEM MEMBERS
    template <int dim>
    MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
    :
    pcout (std::cout),
    dof_handler (triangulation),
    fe (order),
    mpi_communicator (MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator))
    /*
     triangulation (mpi_communicator,
     typename Triangulation<dim>::MeshSmoothing
     (Triangulation<dim>::smoothing_on_refinement |
     Triangulation<dim>::smoothing_on_coarsening)),
     
     dof_handler (triangulation),
     
     //    fe (FE_Nedelec<dim>(order), dim),
     pcout (std::cout),
     mpi_communicator (MPI_COMM_WORLD),
     n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
     this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator))
     */
    {
        p_order = order;
        quad_order = p_order+2;
        pcout.set_condition(this_mpi_process == 0);
    }
    template <int dim>
    MaxwellProblem<dim>::~MaxwellProblem ()
    {
        dof_handler.clear ();
    }
    
    
    
    //computes dot product of 2d or 3d vector
    /*
     template<int dim>
     double MaxwellProblem<dim>::dotprod(const Vector<double> &A, const Vector<double> &B) const
     {
     double return_val = 0;
     for(unsigned int k = 0; k < dim; k++) {
     return_val += A(k)*B(k);
     }
     return return_val;
     }
     */
    template<int dim>
    double MaxwellProblem<dim>::dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const
    {
        double return_val = 0;
        for(unsigned int k = 0; k < dim; k++) {
            return_val += A[k]*B[k];
        }
        return return_val;
    }
    
    //this one exists to interact with boundary values or RHS vectors (which return Vector, not Tensor)
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
        GridTools::partition_triangulation (n_mpi_processes, triangulation);
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::subdomain_wise (dof_handler);
        const types::global_dof_index n_local_dofs
        = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                           this_mpi_process);
        system_matrix.reinit (mpi_communicator,
                              dof_handler.n_dofs(),
                              dof_handler.n_dofs(),
                              n_local_dofs,
                              n_local_dofs,
                              dof_handler.max_couplings_between_dofs());
        solution.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        system_rhs.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
        VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, ExactSolution<dim>(), 0, hanging_node_constraints);
        //         VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, DirichletBoundaryValues<dim>(), 0, hanging_node_constraints);
        hanging_node_constraints.close ();
    }
/* OLD.. probs wrong:
    template <int dim>
    void MaxwellProblem<dim>::setup_system ()
    {
        GridTools::partition_triangulation (n_mpi_processes, triangulation);
        dof_handler.distribute_dofs (fe);
        DoFRenumbering::subdomain_wise (dof_handler);
	
	local_dofs_per_process.resize (n_mpi_processes);
	for (unsigned int i=0; i<n_mpi_processes; ++i)
	{
	  local_dofs_per_process[i] = DoFTools::count_dofs_with_subdomain_association (dof_handler, i);
	}
	
	n_local_dofs = local_dofs_per_process[this_mpi_process];
	
        solution.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
        system_rhs.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);

// Add hanging node constraints and boundary value constraints:
        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
        VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, ExactSolution<dim>(), 0, hanging_node_constraints);
        hanging_node_constraints.close ();
	
	CompressedSparsityPattern sparsity_pattern (dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, hanging_node_constraints, false);
	
	system_matrix.reinit (mpi_communicator,
                              sparsity_pattern,
                              n_local_dofs,
                              n_local_dofs,
                              dof_handler.max_couplings_between_dofs());
    }
 */
    template <int dim>
    void MaxwellProblem<dim>::assemble_system ()
    {
        QGauss<dim>  quadrature_formula(quad_order);
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values   | update_gradients |
                                 update_quadrature_points | update_JxW_values);
        
        FEValuesViews::Vector<dim> fe_views(fe_values, 0);
        
        const unsigned int   dofs_per_cell = fe.dofs_per_cell;
        const unsigned int   n_q_points    = quadrature_formula.size();
        FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       cell_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        RightHandSide<dim>      right_hand_side;
        std::vector<Vector<double> > rhs_values (n_q_points,
                                                 Vector<double>(dim));
        
        Tensor<1,dim> value_i, value_j;
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
            if (cell->subdomain_id() == this_mpi_process)
            {
                cell_matrix = 0;
                cell_rhs = 0;
                fe_values.reinit (cell);
                right_hand_side.vector_value_list (fe_values.get_quadrature_points(),
                                                   rhs_values);
                
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        //const unsigned int component_i = fe.system_to_component_index(i).first;
                        value_i[0] = fe_values.shape_value_component(i,q_point,0);
                        value_i[1] = fe_values.shape_value_component(i,q_point,1);
                        if (dim == 3) {
                            value_i[2] = fe_values.shape_value_component(i,q_point,2);
                        }
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                        {
                            // const unsigned int component_j = fe.system_to_component_index(j).first;
                            value_j[0] = fe_values.shape_value_component(j,q_point,0);
                            value_j[1] = fe_values.shape_value_component(j,q_point,1);
                            if (dim == 3) {
                                value_j[2] = fe_values.shape_value_component(j,q_point,2);
                            }
                            cell_matrix(i,j) += ( fe_views.curl(i,q_point)[0]*fe_views.curl(j,q_point)[0]
                                                 + dotprod(value_i,value_j) )*fe_values.JxW(q_point);
                            
                        }
                        cell_rhs(i) += dotprod(value_i,rhs_values[q_point])*fe_values.JxW(q_point);
                    }
                    
                }
                cell->get_dof_indices (local_dof_indices);
                hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
                
            }
        //        pcout << "Printing matrix" << std::endl;
        //        cell = dof_handler.begin_active();
        //        for (; cell!=endc; ++cell)
        //        {
        //        pcout << "cell matrix" << std::endl;
        //            for (unsigned int i=0; i<dofs_per_cell; ++i)
        //            {
        //                for (unsigned int j=0; j<dofs_per_cell; ++j)
        //                {
        //                    pcout << cell_matrix(i,j) << " ";
        //                }
        //                pcout << std::endl;
        //            }
        //        }
        
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
        
        // Cannot use interpolate_boundary_values for Nedelec elements.
        //	VectorTools::interpolate_boundary_values (dof_handler,
        //						  0,
        //					   ZeroFunction<dim>(dim),
        //						  boundary_values);
        /*
         VectorTools::project_boundary_values_curl_conforming (dof_handler, //	const DoFHandler< dim > & 	dof_handler,
         0, //const unsigned int 	first_vector_component,
         boundary_function, //const Function< dim > & 	boundary_function,
         boundary_component, //const types::boundary_id 	boundary_component,
         constraints, //ConstraintMatrix & 	constraints,
         boundary_values //const Mapping< dim > & 	mapping = StaticMappingQ1< dim >::mapping
         );
         
         
         
         MatrixTools::apply_boundary_values (boundary_values,
         system_matrix, solution,
         system_rhs, false);
         */
        
    }
    /*
    template <int dim>
    unsigned int MaxwellProblem<dim>::solve ()
    {
        SolverControl           solver_control (solution.size(),
                                                1e-12*system_rhs.l2_norm());
        PETScWrappers::SolverCG cg (solver_control,
                                    mpi_communicator);
        PETScWrappers::PreconditionNone preconditioner(system_matrix);
        cg.solve (system_matrix, solution, system_rhs, preconditioner);
        PETScWrappers::Vector localized_solution (solution);
        hanging_node_constraints.distribute (localized_solution);
        solution = localized_solution;
        return solver_control.last_step();
    }
    */
    template <int dim>
    unsigned int MaxwellProblem<dim>::solve ()
    {
        SolverControl           solver_control (solution.size(),
                                                1e-10*system_rhs.l2_norm());
        PETScWrappers::SparseDirectMUMPS cg (solver_control,
                                             mpi_communicator);
        PETScWrappers::PreconditionNone preconditioner(system_matrix);
        cg.solve (system_matrix, solution, system_rhs
                  );
        PETScWrappers::Vector localized_solution (solution);
        hanging_node_constraints.distribute (localized_solution);
        solution = localized_solution;
        return solver_control.last_step();
    }
    template<int dim>
    void MaxwellProblem<dim>::process_solution(const unsigned int cycle)
    {
        const ExactSolution<dim> exact_solution;
        //const BesselSolution<dim> exact_solution(2.0/3.0,1.0);
        
        //        Vector<double> local_errors (triangulation.n_active_cells());
        //        VectorTools::integrate_difference(dof_handler, solution, exact_solution, local_errors, QGauss<dim>(quad_order), VectorTools::L2_norm);
        //        const double total_local_error = local_errors.l2_norm();
        //        const double total_global_error = std::sqrt (Utilities::MPI::sum (total_local_error * total_local_error, MPI_COMM_WORLD));
        //        convergence_table.add_value("cycle", cycle);
        //        convergence_table.add_value("cells", triangulation.n_active_cells());
        //        convergence_table.add_value("dofs", dof_handler.n_dofs());
        //        convergence_table.add_value("L2 Error", total_global_error);
        
        const PETScWrappers::Vector localized_solution (solution);
        //        if (this_mpi_process == 0)
        //        {
        Vector<double> diff_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference(dof_handler, localized_solution, exact_solution,
                                          diff_per_cell, QGauss<dim>(quad_order), VectorTools::L2_norm);
        const double L2_error = diff_per_cell.l2_norm();
        
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());
        convergence_table.add_value("L2 Error", L2_error);
        
        /* H1 norm
         VectorTools::integrate_difference (dof_handler, localized_solution, exact_solution,
         diff_per_cell, QGauss<dim>(quad_order), VectorTools::H1_seminorm);
         const double H1_error = diff_per_cell.l2_norm();
         convergence_table.add_value("H1 Error", H1_error);
         */
        
    }
    
    template <int dim>
    void MaxwellProblem<dim>::refine_grid ()
    {
        const PETScWrappers::Vector localized_solution (solution);
        Vector<float> local_error_per_cell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(quad_order),
                                            typename FunctionMap<dim>::type(),
                                            localized_solution,
                                            local_error_per_cell,
                                            ComponentMask(),
                                            0,
                                            multithread_info.n_threads(),
                                            this_mpi_process);
        const unsigned int n_local_cells
        = GridTools::count_cells_with_subdomain_association (triangulation,
                                                             this_mpi_process);
        PETScWrappers::MPI::Vector
        distributed_all_errors (mpi_communicator,
                                triangulation.n_active_cells(),
                                n_local_cells);
        for (unsigned int i=0; i<local_error_per_cell.size(); ++i)
            if (local_error_per_cell(i) != 0)
                distributed_all_errors(i) = local_error_per_cell(i);
        distributed_all_errors.compress (VectorOperation::insert);
        const Vector<float> localized_all_errors (distributed_all_errors);
        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         localized_all_errors,
                                                         0.3, 0.03);
        triangulation.execute_coarsening_and_refinement ();
    }
    
    template <int dim>
    void MaxwellProblem<dim>::output_results_eps (const unsigned int cycle) const
    {
        const PETScWrappers::Vector localized_solution (solution);
        if (this_mpi_process == 0)
        {
            
            DataOut<dim> data_out;
            data_out.attach_dof_handler (dof_handler);
            data_out.add_data_vector (localized_solution[1], "solution");
            data_out.build_patches ();
            DataOutBase::EpsFlags eps_flags;
            eps_flags.z_scaling = 1;
            eps_flags.azimut_angle = 0;
            eps_flags.turn_angle   = 0;
            data_out.set_flags (eps_flags);
            std::ostringstream filename;
            filename << "solution-" << cycle << ".eps";
            std::ofstream output (filename.str().c_str());
            data_out.write_eps (output);
        }
    }
    template <int dim>
    void MaxwellProblem<dim>::output_results_vtk (const unsigned int cycle) const
    {
        const PETScWrappers::Vector localized_solution (solution);
        if (this_mpi_process == 0)
        {
            std::ostringstream filename;
            filename << "solution-" << cycle << ".vtk";
            std::ofstream output (filename.str().c_str());
            DataOut<dim> data_out;
            data_out.attach_dof_handler (dof_handler);
            std::vector<std::string> solution_names;
            switch (dim)
            {
                case 1:
                    solution_names.push_back ("E");
                    break;
                case 2:
                    solution_names.push_back ("E1");
                    solution_names.push_back ("E2");
                    break;
                case 3:
                    solution_names.push_back ("E1");
                    solution_names.push_back ("E2");
                    solution_names.push_back ("E3");
                    break;
                default:
                    Assert (false, ExcNotImplemented());
            }
            data_out.add_data_vector (localized_solution, solution_names);
            std::vector<unsigned int> partition_int (triangulation.n_active_cells());
            GridTools::get_subdomain_association (triangulation, partition_int);
            const Vector<double> partitioning(partition_int.begin(),
                                              partition_int.end());
            data_out.add_data_vector (partitioning, "partitioning");
            data_out.build_patches (p_order+2);
            data_out.write_vtk (output);
        }
    }
    template <int dim>
    void MaxwellProblem<dim>::output_results_gmv(const unsigned int cycle) const
    {
        const PETScWrappers::Vector localized_solution (solution);
        if (this_mpi_process == 0)
        {
            std::ostringstream filename;
            filename << "solution-" << cycle << ".gmv";
            std::ofstream output (filename.str().c_str());
            DataOut<dim> data_out;
            data_out.attach_dof_handler (dof_handler);
            std::vector<std::string> solution_names;
            switch (dim)
            {
                case 1:
                    solution_names.push_back ("E");
                    break;
                case 2:
                    solution_names.push_back ("E1");
                    solution_names.push_back ("E2");
                    break;
                case 3:
                    solution_names.push_back ("E1");
                    solution_names.push_back ("E2");
                    solution_names.push_back ("E3");
                    break;
                default:
                    Assert (false, ExcNotImplemented());
            }
            data_out.add_data_vector (localized_solution, solution_names);
            std::vector<unsigned int> partition_int (triangulation.n_active_cells());
            GridTools::get_subdomain_association (triangulation, partition_int);
            const Vector<double> partitioning(partition_int.begin(),
                                              partition_int.end());
            data_out.add_data_vector (partitioning, "partitioning");
            data_out.build_patches (p_order+2);
            data_out.write_gmv (output);
        }
    }
    template <int dim>
    void MaxwellProblem<dim>::run ()
    {
        for (unsigned int cycle=0; cycle<6; ++cycle)
        {
            pcout << "Cycle " << cycle << ':' << std::endl;
            if (cycle == 0)
            {
                GridGenerator::hyper_cube (triangulation, -1, 1);
                triangulation.refine_global (1);
            }
            else
                //refine_grid ();
                triangulation.refine_global(1); // ignores more complicated grid.
            pcout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
            setup_system ();
            pcout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << " (by partition:";
            for (unsigned int p=0; p<n_mpi_processes; ++p)
                pcout << (p==0 ? ' ' : '+')
                << (DoFTools::
                    count_dofs_with_subdomain_association (dof_handler,
                                                           p));
            pcout << ")" << std::endl;
            assemble_system ();
            const unsigned int n_iterations = solve ();
            pcout << "   Solver converged in " << n_iterations << " iterations." << std::endl;
            process_solution (cycle);
            //   output_results_gmv (cycle);
        }
        
        pcout << std::endl;
        if (this_mpi_process == 0)
        {
            convergence_table.set_precision("L2 Error",12);
            convergence_table.set_scientific("L2 Error",true);
            /*
            convergence_table.set_precision("H1 Error",12);
            convergence_table.set_scientific("H1 Error",true);
             */
            convergence_table.write_text(std::cout);

        }
        
        
    }
    
}
// END MAXWELL PROBLEM MEMBERS


// MAIN FUNCTION
int main (int argc, char **argv)
{
    try
    {
        using namespace dealii;
        using namespace Maxwell_FEM;
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        {
            deallog.depth_console (0);
            MaxwellProblem<2> maxwell_problem(5);
            maxwell_problem.run ();
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    return 0;
}