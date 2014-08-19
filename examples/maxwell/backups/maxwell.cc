#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/thread_management.h>		//for multithreading! Go, i7, go!
#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
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
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <fstream>
#include <iostream>
#include <math.h>

namespace MaxwellFEM
{
    using namespace dealii;
    
    template<int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution() : Function<dim>() {}
        virtual double value (const Point<dim> &p, const unsigned int component) const;
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
        virtual Tensor<1,dim> gradient(const Point<dim> &p, const unsigned int component) const;
    };
    
    template<int dim>
    double ExactSolution<dim>::value(const Point<dim> &p, const unsigned int component) const
    {
        Assert (dim >= 2, ExcNotImplemented());
        Assert (component > dim-1, ExcNotImplemented());
        
        //second 2D example from Anna
        const double PI = acos(-1);
        double val = -1000;
        switch(component) {
            case 0:		val = cos(PI*p(0))*sin(PI*p(1));
            case 1: 	val = -sin(PI*p(0))*cos(PI*p(1));
        }
        return val;
    }
    
    template<int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
    {
        Assert(dim >= 2, ExcNotImplemented());
        const double PI = acos(-1);
        values(0) = cos(PI*p(0))*sin(PI*p(1));
        values(1) = -sin(PI*p(0))*cos(PI*p(1));
    }
    
    template <int dim>
    Tensor<1,dim> ExactSolution<dim>::gradient(const Point<dim> &p, const unsigned int component) const
    {
        Assert (dim >= 2, ExcNotImplemented());
        Assert (component > dim-1, ExcNotImplemented());
        
        Tensor<1,dim> value;
        
        const double PI = acos(-1);
        switch(component) {
            case 0:		value[0] = -PI*sin(PI*p(0))*sin(PI*p(1));
                value[1] = PI*cos(PI*p(0))*cos(PI*p(1));
            case 1:		value[0] = -PI*cos(PI*p(0))*cos(PI*p(1));
                value[1] = PI*sin(PI*p(0))*sin(PI*p(1));
        }
        return value;
    }
    
    template <int dim>
    class MaxwellToy
    {
    public:
        MaxwellToy (const unsigned int order);
        ~MaxwellToy ();
        void run (const unsigned int maxcycles);
        
    private:
        //vector stuff
        double dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const;
        double dotprod(const Tensor<1,dim> &A, const Vector<double> &B) const;
        
        //finite element stuff
        void setup_system ();
        void assemble_system ();
        void assemble_system_chunk(const typename DoFHandler<dim>::active_cell_iterator &begin, const typename DoFHandler<dim>::active_cell_iterator &end);
        void solve ();
        void process_solution(const unsigned int cycle);
        void output_results (const unsigned int cycle) const;
        
        //member variables
        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dof_handler;
        FE_Nedelec<dim>      fe;
        
        ConstraintMatrix     hanging_node_constraints;
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;
        
        Vector<double>       solution;
        Vector<double>       system_rhs;
        
        ConvergenceTable	   convergence_table;
        Threads::ThreadMutex assembler_lock;
    };
    
    
    //<-------------- RIGHT HAND SIDE CLASS --------------->
    template <int dim>
    class RightHandSide :  public Function<dim>
    {
    public:
        RightHandSide ();
        
        virtual void vector_value (const Point<dim> &p, Vector<double> &values) const;
        virtual void vector_value_list (const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const;
    };
    
    
    template <int dim>
    RightHandSide<dim>::RightHandSide ()
    :
    Function<dim> (dim)
    {}
    
    
    template <int dim>
    inline
    void RightHandSide<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        Assert (values.size() == dim, ExcDimensionMismatch (values.size(), dim));
        Assert (dim >= 2, ExcNotImplemented());
        
        //second 2D example from Anna
        const double PI = acos(-1);
        values(0) = (2*PI*PI + 1)*cos(PI*p(0))*sin(PI*p(1));
        values(1) = -(2*PI*PI + 1)*sin(PI*p(0))*cos(PI*p(1));
    }
    
    template <int dim>
    void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const
    {
        Assert (value_list.size() == points.size(), ExcDimensionMismatch (value_list.size(), points.size()));
        
        const unsigned int n_points = points.size();
        
        for (unsigned int p=0; p<n_points; ++p) {
            RightHandSide<dim>::vector_value (points[p], value_list[p]);
        }
    }
    
    
    //<-------------- BOUNDARY VALUE CLASS --------------->
    
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        BoundaryValues ();
        virtual double value(const Point<dim> &p) const;
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
        virtual void vector_value_list(const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const;
    };
    
    template <int dim>
    BoundaryValues<dim>::BoundaryValues ()
    :
    Function<dim> (dim)
    {}
    
    //this can be used for 2D BCs, since we only have one component to deal with
    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> &/*p*/) const
    {
        return 0;
    }
    
    template <int dim>
    inline
    void BoundaryValues<dim>::vector_value (const Point<dim> &/*p*/, Vector<double> &values) const
    {
      values(0) = 0;
      values(1) = 0;
    }
    
    template <int dim>
    void BoundaryValues<dim>::vector_value_list (const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const
    {
        Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
        
        const unsigned int n_points = points.size();
        
        for (unsigned int p=0; p<n_points; p++) {
            BoundaryValues<dim>::vector_value(points[p],value_list[p]);
        }
    }
    
    //
    //<-------------- BEGIN MAXWELL CLASS ----------------->
    //
    
    template <int dim>
    MaxwellToy<dim>::MaxwellToy (const unsigned int order)
    :
    dof_handler (triangulation),
    fe(order)
    {}
    
    
    
    
    template <int dim>
    MaxwellToy<dim>::~MaxwellToy ()
    {
        dof_handler.clear ();
    }
    
    
    //computes dot product of 2d or 3d vector
    template<int dim>
    double MaxwellToy<dim>::dotprod(const Tensor<1,dim> &A, const Tensor<1,dim> &B) const
    {
        double return_val = 0;
        for(unsigned int k = 0; k < dim; k++) {
            return_val += A[k]*B[k];
        }
        return return_val;
    }
    
    //this one exists to interact with boundary values or RHS vectors (which return Vector, not Tensor)
    template<int dim>
    double MaxwellToy<dim>::dotprod(const Tensor<1,dim> &A, const Vector<double> &B) const
    {
        double return_val = 0;
        for(unsigned int k = 0; k < dim; k++) {
            return_val += A[k]*B(k);
        }
        return return_val;
    }
    
    
    template <int dim>
    void MaxwellToy<dim>::setup_system ()
    {
        dof_handler.distribute_dofs (fe);
        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
        hanging_node_constraints.close ();
        sparsity_pattern.reinit (dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.max_couplings_between_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
        hanging_node_constraints.condense (sparsity_pattern);
        sparsity_pattern.compress();
        
        system_matrix.reinit (sparsity_pattern);
        solution.reinit (dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
        
        std::cout << " (" << dof_handler.n_dofs() << " DoFs)...";
    }
    
    
    template <int dim>
    void MaxwellToy<dim>::assemble_system()
    {
        const unsigned int n_threads = multithread_info.n_cpus;
        Threads::ThreadGroup<> threads;
        
        typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
        std::vector<std::pair<active_cell_iterator, active_cell_iterator> > thread_ranges = Threads::split_range<active_cell_iterator>(dof_handler.begin_active(), dof_handler.end(), n_threads);
        for (unsigned int thread=0; thread < n_threads; thread++) {
            threads += Threads::new_thread(&MaxwellToy<dim>::assemble_system_chunk, *this, thread_ranges[thread].first, thread_ranges[thread].second);
        }
        threads.join_all();
        
        hanging_node_constraints.condense(system_matrix);
        hanging_node_constraints.condense(system_rhs);
        
        BoundaryValues<dim> boundary_function;
        //this should be capable of handling non-homogenous Dirichlet BCs
        //REMEMBER: interpolate_boundary_values does NOT work with Nedelec elements!!
        std::map<unsigned int,double> boundary_values;
        double diameter;
        Point<dim> midpoint;
        unsigned int boundary_dof;
        
        typename DoFHandler<dim>::active_cell_iterator bcell = dof_handler.begin_active(), endbc = dof_handler.end();
        
        for(; bcell!=endbc; bcell++) {
            if(bcell -> at_boundary()) {
                for(unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; face++) {
                    if(bcell->face(face)->at_boundary()) {
                        //the (only!) Nedelec-DoF on the current boundary-edge e, is a line integral over e.  We approximate it with the midpoint rule
                        //(A quadrature rule of order 1 is sufficient not to affect order of convergence of this FEM)
                        midpoint = bcell -> face(face) -> center();
                        diameter = bcell -> face(face) -> diameter();
                        
                        //set the (only!) DoF on the current boundary-edge e to the value |e|*boundary_function(midpoint)
                        boundary_dof = bcell -> face(face) -> dof_index(0);
                        boundary_values[boundary_dof] = diameter * boundary_function.value(midpoint);
                    }
                }
            }
        }
        MatrixTools::apply_boundary_values (boundary_values, system_matrix, solution, system_rhs);
    }
    
    
    template <int dim>
    void MaxwellToy<dim>::assemble_system_chunk(const typename DoFHandler<dim>::active_cell_iterator &begin, const typename DoFHandler<dim>::active_cell_iterator &end)
    {
        QGauss<dim>  quadrature_formula(4);
        FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        
        FEValuesViews::Vector<dim> fe_views(fe_values, 0);				//FEValues inherits FEValuesBase, so it's ok to pass it here
        
        FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs (dofs_per_cell);
        
        std::vector<unsigned int> local_dof_indices (dofs_per_cell);
        
        RightHandSide<dim>      right_hand_side;
        std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(dim));
        
        //this is for storing stuff inside the (i,j) loops; here for memory purposes
        Tensor<1,dim> value_i, value_j;
        
        typename DoFHandler<dim>::active_cell_iterator cell;
        for (cell=begin; cell!=end; ++cell) {
            cell_matrix = 0;
            cell_rhs = 0;
            fe_values.reinit (cell);
            right_hand_side.vector_value_list (fe_values.get_quadrature_points(), rhs_values);
            
            for(unsigned int q_point=0; q_point < n_q_points; q_point++) {
                for (unsigned int i=0; i<dofs_per_cell; ++i) {
                    value_i[0] = fe_values.shape_value_component(i,q_point,0);
                    value_i[1] = fe_values.shape_value_component(i,q_point,1);
                    if (dim == 3) {
                        value_i[2] = fe_values.shape_value_component(i,q_point,2);
                    }
                    
                    for (unsigned int j=0; j<dofs_per_cell; ++j) {
                        value_j[0] = fe_values.shape_value_component(j,q_point,0);
                        value_j[1] = fe_values.shape_value_component(j,q_point,1);
                        if (dim == 3) {
                            value_j[2] = fe_values.shape_value_component(j,q_point,2);
                        }
                        
                        cell_matrix(i,j) += (fe_views.curl(i,q_point)[0]*fe_views.curl(j,q_point)[0] + dotprod(value_i,value_j))*fe_values.JxW(q_point);
                    }
                }
            }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i) {
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
                    value_i[0] = fe_values.shape_value_component(i,q_point,0);
                    value_i[1] = fe_values.shape_value_component(i,q_point,1);
                    if(dim == 3) {
                        value_i[2] = fe_values.shape_value_component(i,q_point,2);
                    }
                    cell_rhs(i) += dotprod(value_i, rhs_values[q_point])*fe_values.JxW(q_point);
                }
            }
            
            assembler_lock.acquire();
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i) {
                for (unsigned int j=0; j<dofs_per_cell; ++j) {
                    system_matrix.add (local_dof_indices[i], local_dof_indices[j], cell_matrix(i,j));
                }
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
            assembler_lock.release();
        }
    }
    
    
    template <int dim>
    void MaxwellToy<dim>::solve ()
    {
        SolverControl           solver_control (10000, 1e-12);
        SolverCG<>	    	    cg_solver(solver_control);
        
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.0);
        
        cg_solver.solve (system_matrix, solution, system_rhs, preconditioner);
        
        hanging_node_constraints.distribute (solution);
    }
    
    
    template<int dim>
    void MaxwellToy<dim>::process_solution(const unsigned int cycle)
    {
        const ExactSolution<dim> exact_solution;
        
        Vector<double> diff_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference(dof_handler, solution, exact_solution, diff_per_cell, QGauss<dim>(4), VectorTools::L2_norm);
        const double L2_error = diff_per_cell.l2_norm();
        
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());
        convergence_table.add_value("L2 Error", L2_error);
    }
    
    
    template <int dim>
    void MaxwellToy<dim>::run (const unsigned int maxcycles)
    {
        std::cout << "Multithreading Enabled! (Using maximum number of CPUs = " << multithread_info.n_cpus << ")\n";
        std::cout << "\n\n";
        
        for (unsigned int cycle=0; cycle<maxcycles; ++cycle)
        {
            std::cout << "Running cycle #" << cycle;
            if (cycle == 0) {
                GridGenerator::hyper_cube (triangulation, -1, 1);
                triangulation.refine_global (2);
            }
            else
                triangulation.refine_global(1);
            
            //This line appears in setup_system, when the DoFs are actually correct
            //std::cout << " (" << dof_handler.n_dofs() << " DoFs)...";
            
            setup_system();			std::cout << "...";
            assemble_system ();		std::cout << "......";
            solve ();			std::cout << "......";
            process_solution (cycle);	std::cout << "Done!\n";
        }
        
        convergence_table.set_precision("L2 Error",3);
        convergence_table.set_scientific("L2 Error",true);
        
        std::cout << std::endl;
        convergence_table.write_text(std::cout);
    }
    
}

int main ()
{
    unsigned int order = 0;
    unsigned int maxcycles = 10;
    
    try
    {
        dealii::deallog.depth_console (0);
        
        MaxwellFEM::MaxwellToy<2> maxwelltoy(order);
        maxwelltoy.run (maxcycles);
    }
    catch (std::exception &exc)
    {
        std::cerr << "\n\n----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: \n" << exc.what() << "\nAborting!\n" << "----------------------------------------------------" << std::endl;
        
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n\n----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!\n" << "Aborting!\n" << "----------------------------------------------------" << std::endl;
        return 1;
    }
    
    return 0;
}