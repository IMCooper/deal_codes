/*
 Solves curl(curl(u)) + u = f in 2D
 
 exact solution is given by:
 u(0) = cos(pi*x)*sin(pi*y) + C
 u(1) = -sin(pi*x)*cos(pi(y) + C
 
 f(0) = (2*pi^2 + 1)*cos(pi*x)*sin(pi*x) + C
 f(1) = -(2*pi^2 + 1)*sin(pi*x)*cos(pi(y) + C
 
 Other option is Bessel solution (singularity at l-shape corner)
 
 where C is some constant. To change C, edit bc_constant in ExactSolution and RightHandSide classes.
 
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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
// Req'd for bessel functions in L-shape solution w/ singularity.
#include <boost/math/special_functions/bessel.hpp>

using namespace dealii;
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
    //void read_in_mesh (std::string mesh_file);
    void setup_system ();
    void assemble_system ();
    void solve ();
    void process_solution(const unsigned int cycle);
    void output_results_eps (const unsigned int cycle) const;
    void output_results_vtk (const unsigned int cycle) const;
    void output_results_gmv(const unsigned int cycle) const;
    //void mesh_info(const Triangulation<dim> &tria, const std::string        &filename);
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FE_Nedelec<dim>            fe;
    ConstraintMatrix     constraints;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;
    
    unsigned int p_order;
    unsigned int quad_order;
    
    // Constants:
    const double constant_PI=dealii::numbers::PI;
    const double constant_epsilon0=8.85418782e-12; // electric constant (permittivity)
    const double constant_mu0=;1.25663706e-6; // magnetic constant (permeability)
    const double constant_sigma0=1e-5; // background conductivity, set to regularise system.
    // Material Parameters:
    double param_omega = 100; // angular frequency, rads/sec. Set elsewhere later.
    double param_mu_star = 2e6; // permability in conducting region. Will be set elsewhere later.
    Vector<double> param_mur; 
    Vector<double> param_kappa; // = -omega.^2*epr + i*omega*sigma
    
    ConvergenceTable	   convergence_table;
};
// EXACT SOLUTION CLASS
template<int dim>
class ExactSolution : public Function<dim>
{
public:
    ExactSolution() : Function<dim>(dim) {}
    virtual double value (const Point<dim> &p,
                          const unsigned int component) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double> &result) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &values) const;
private:
    const double PI = dealii::numbers::PI;
	const double bc_constant = 0.1;
};
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
	const double bc_constant = 0.1;
};
template<int dim>
// BESSEL SOLUTION CLASS. USE FOR L-SHAPE WITH SINGULARITY.
class BesselSolution : public Function<dim>
{
public:
    BesselSolution (double alp, double omg);
    virtual double value (const Point<dim> &p, const unsigned int component) const;
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
    virtual void value_list (const std::vector<Point<dim> > &points, std::vector<double> &values, const unsigned int component) const;
    virtual void vector_value_list(const std::vector<Point<dim> > &points, std::vector<Vector<double> > &value_list) const;
private:
    double alpha;
    double omega;
    const double PI = dealii::numbers::PI;
};
// DEFINE EXACT SOLUTION MEMBERS
template<int dim>
double ExactSolution<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const
{
    Assert (dim >= 2, ExcNotImplemented());
    AssertIndexRange(component, dim);
    
    double val = -1000;
    switch(component) {
        case 0:	val = cos(PI*p(0))*sin(PI*p(1)) + bc_constant;
        case 1:	val = -sin(PI*p(0))*cos(PI*p(1)) + bc_constant;
        case 2: val = 1;
    }
    return val;
    
}
template<int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &result) const
{
    Assert(dim >= 2, ExcNotImplemented());
    result(0) = cos(PI*p(0))*sin(PI*p(1)) + bc_constant;
    result(1) = -sin(PI*p(0))*cos(PI*p(1)) + bc_constant;
    if (dim == 3) {
        result(2)=1;
    }
    
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
                values[i] = cos(PI*p(0))*sin(PI*p(1)) + bc_constant;
            case 1:
                values[i] = -sin(PI*p(0))*cos(PI*p(1)) + bc_constant;
            case 2:
                values[i] = 1;
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
        values[i](0) = cos(PI*p(0))*sin(PI*p(1)) + bc_constant;
        values[i](1) = -sin(PI*p(0))*cos(PI*p(1)) + bc_constant;
        if (dim == 3) {
            values[i](2) = 1;
        }
    }
}
// END EXACT SOLUTION MEMBERS

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
    values(0) = (2*PI*PI + 1)*cos(PI*p(0))*sin(PI*p(1)) + bc_constant;
    values(1) = -(2*PI*PI + 1)*sin(PI*p(0))*cos(PI*p(1)) + bc_constant;
    if (dim==3) {
        values(2) = 1;
    }
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

// BESSEL SOLUTION MEMBERS
template <int dim>
BesselSolution<dim>::BesselSolution (double alp, double omg)
:
Function<dim> (dim)
{
    alpha=alp;
    omega=omg;
}

template<int dim>
double BesselSolution<dim>::value(const Point<dim> &p,
                                  unsigned int component) const
{
    Assert (dim >= 2, ExcNotImplemented());
    AssertIndexRange(component, dim);
    
    double val = -1000;
    
    double r;
    double theta;
    double ja;
    double jm1;
    double jp1;
    double dja;
    double jp2;
    double jm2;
    double ddja;
    double dhdr;
    double dhdt;
    double drdx;
    double drdy;
    double dtdx;
    double dtdy;
    
    double xp=p(0);
    double yp=p(1);
    
    r=sqrt(xp*xp + yp*yp);
    theta=atan2(yp,xp);
    if (std::abs(theta) < 1e-6)
    {
        theta=0;
    }
    theta=theta+PI/2.0;
    ja=boost::math::cyl_bessel_j(alpha,omega*r);
    
    jm1=(((2.0*alpha)/(omega*r))*boost::math::cyl_bessel_j(alpha,omega*r))-boost::math::cyl_bessel_j(alpha+1,omega*r);
    jp1=boost::math::cyl_bessel_j(alpha+1,omega*r);
    dja=0.5*(jm1-jp1);
    jp2=boost::math::cyl_bessel_j(alpha+2,omega*r);
    jm2=((2.0*(-1.0+alpha)/(omega*r))*jm1)-ja;
    
    ddja=0.5*((0.5*(jm2-ja))+(0.5*(jp2-ja)));
    //   first order derivitives
    dhdr=omega*dja*(cos(alpha*theta));
    dhdt=-1.0*alpha*ja*(sin(alpha*theta));
    drdx=xp/r;
    drdy=yp/r;
    
    dtdx=-yp/(r*r);
    dtdy=xp/(r*r);
    
    switch (component)
    {
        case 0:
            val =  (dhdr*drdy)+(dhdt*dtdy);
            break;
        case 1:
            val = -(dhdr*drdx)-(dhdt*dtdx);
            break;
    }
    return val;
}

template <int dim>
inline
void BesselSolution<dim>::vector_value (const Point<dim> &p,
                                        Vector<double> &values) const
{
    Assert (values.size() == 2, ExcDimensionMismatch (values.size(), 2));
    
    double r;
    double theta;
    double ja;
    double jm1;
    double jp1;
    double dja;
    double jp2;
    double jm2;
    double ddja;
    double dhdr;
    double dhdt;
    double drdx;
    double drdy;
    double dtdx;
    double dtdy;
    
    double xp=p(0);
    double yp=p(1);
    
    r=sqrt(pow(xp,2)+pow(yp,2));
    theta=atan2(yp,xp);
    if (std::abs(theta) < 1e-6)
    {
        theta=0;
    }
    theta=theta+PI/2.0;
    ja=boost::math::cyl_bessel_j(alpha,omega*r);
    
    jm1=(((2.0*alpha)/(omega*r))*boost::math::cyl_bessel_j(alpha,omega*r))-boost::math::cyl_bessel_j(alpha+1,omega*r);
    jp1=boost::math::cyl_bessel_j(alpha+1,omega*r);
    dja=0.5*(jm1-jp1);
    jp2=boost::math::cyl_bessel_j(alpha+2,omega*r);
    jm2=((2.0*(-1.0+alpha)/(omega*r))*jm1)-ja;
    
    ddja=0.5*((0.5*(jm2-ja))+(0.5*(jp2-ja)));
    //   first order derivitives
    dhdr=omega*dja*(cos(alpha*theta));
    dhdt=-1.0*alpha*ja*(sin(alpha*theta));
    
    drdx=xp/r;
    drdy=yp/r;
    
    dtdx = -yp/(r*r);
    dtdy = xp/(r*r);
    values(0) =  (dhdr*drdy)+(dhdt*dtdy);
    values(1) = -(dhdr*drdx)-(dhdt*dtdx);
}

template <int dim>
void BesselSolution<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double> &values,
                                      const unsigned int component) const
{
    Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    AssertIndexRange(component, dim);
    
    double r;
    double theta;
    double ja;
    double jm1;
    double jp1;
    double dja;
    double jp2;
    double jm2;
    double ddja;
    double dhdr;
    double dhdt;
    double drdx;
    double drdy;
    double dtdx;
    double dtdy;
    double xp;
    double yp;
    
    for (unsigned int i=0; i<points.size(); ++i)
    {
        const Point<dim> &p = points[i];
        xp=p(0);
        yp=p(1);
        
        r=sqrt(pow(xp,2)+pow(yp,2));
        theta=atan2(yp,xp);
        if (std::abs(theta) < 1e-6)
        {
            theta=0;
        }
        theta=theta+PI/2.0;
        ja=boost::math::cyl_bessel_j(alpha,omega*r);
        
        jm1=(((2.0*alpha)/(omega*r))*boost::math::cyl_bessel_j(alpha,omega*r))-boost::math::cyl_bessel_j(alpha+1,omega*r);
        jp1=boost::math::cyl_bessel_j(alpha+1,omega*r);
        dja=0.5*(jm1-jp1);
        jp2=boost::math::cyl_bessel_j(alpha+2,omega*r);
        jm2=((2.0*(-1.0+((double)alpha))/(omega*r))*jm1)-ja;
        
        ddja=0.5*((0.5*(jm2-ja))+(0.5*(jp2-ja)));
        //   first order derivitives
        dhdr=omega*dja*(cos(alpha*theta));
        dhdt=-1.0*alpha*ja*(sin(alpha*theta));
        drdx=xp/r;
        drdy=yp/r;
        
        dtdx=-yp/(r*r);
        dtdy=xp/(r*r);
        switch(component)
        {
            case 0:
                values[i] = (dhdr*drdy)+(dhdt*dtdy);
                break;
            case 1:
                values[i] = -(dhdr*drdx)-(dhdt*dtdx);
                break;
        }
    }
}

template <int dim>
void BesselSolution<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    for (unsigned int p=0; p<n_points; p++) {
        BesselSolution<dim>::vector_value(points[p],value_list[p]);
    }
}
// END BESSELSOLUTION MEMBERS

template <int dim>
MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
:
dof_handler (triangulation),
fe (order)
{
    p_order = order;
    quad_order = p_order+1;
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
/*
template <int dim>
void MaxwellProblem<dim>::read_in_mesh (std::string mesh_file)
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(mesh_file);
  gridin.read_msh(f);
  mesh_info(triangulation, "mesh.eps");
}
*/
template <int dim>
void MaxwellProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, ExactSolution<dim>(), 0, constraints);
    
    constraints.close ();
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,false);
    
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit (sparsity_pattern);
    
    // Setup material parameters (could be expanded into own routine:
    param_mur.reinit(2);
    param_mur(0)=1;
    param_mur(1)=param_mu_star/constant_mu0;
    param_kappa.reinit(2);
    param_kappa(0)=;
    param_kappa(1)=;
}
template <int dim>
void MaxwellProblem<dim>::assemble_system ()
{
    const QGauss<dim>  quadrature_formula(quad_order);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
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
                value_i[0] = fe_values.shape_value_component(i,q_point,0);
                value_i[1] = fe_values.shape_value_component(i,q_point,1);
                if (dim == 3) {
                    value_i[2] = fe_values.shape_value_component(i,q_point,2);
                }
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    value_j[0] = fe_values.shape_value_component(j,q_point,0);
                    value_j[1] = fe_values.shape_value_component(j,q_point,1);
                    if (dim == 3) {
                        value_j[2] = fe_values.shape_value_component(j,q_point,2);
                    }
                    /*
                     cell_matrix(i,j) += ( fe_views.curl(i,q_point)[0]*fe_views.curl(j,q_point)[0]
                     + dotprod(value_i,value_j) )*fe_values.JxW(q_point);
                     */
                    cell_matrix(i,j) += ( dotprod(fe_views.curl(i,q_point),fe_views.curl(j,q_point))
                                         + dotprod(value_i,value_j) )*fe_values.JxW(q_point);
                    
                }
                cell_rhs(i) += dotprod(value_i,rhs_values[q_point])*fe_values.JxW(q_point);
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
    /* CG:
     SolverControl           solver_control (1000, 1e-8);
     SolverCG<>              solver (solver_control);
     PreconditionSSOR<> preconditioner;
     preconditioner.initialize(system_matrix, 1.2);
     solver.solve (system_matrix, solution, system_rhs,
     preconditioner);
     constraints.distribute (solution);
     */
    /* GMRES:
     SolverControl           solver_control (1000, 1e-6);
     SolverGMRES<>              solver (solver_control);
     PreconditionSSOR<> preconditioner;
     preconditioner.initialize(system_matrix, 1.0);
     solver.solve (system_matrix, solution, system_rhs,
     preconditioner);
     constraints.distribute (solution);
     */
    /* Direct */
    SparseDirectUMFPACK A_direct;
    //SparseDirectMUMPS A_direct;
    A_direct.initialize(system_matrix);
    
    A_direct.vmult (solution, system_rhs);
    constraints.distribute (solution);
    
}
template<int dim>
void MaxwellProblem<dim>::process_solution(const unsigned int cycle)
{
    const ExactSolution<dim> exact_solution;
    Vector<double> diff_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell, QGauss<dim>(quad_order), VectorTools::L2_norm);
    const double L2_error = diff_per_cell.l2_norm();
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("L2 Error", L2_error);
}
template <int dim>
void MaxwellProblem<dim>::output_results_eps (const unsigned int cycle) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution[1], "solution");
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
template <int dim>
void MaxwellProblem<dim>::output_results_vtk (const unsigned int cycle) const
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
    data_out.add_data_vector (solution, solution_names);
    data_out.build_patches (p_order+2);
    data_out.write_vtk (output);
}
template <int dim>
void MaxwellProblem<dim>::output_results_gmv(const unsigned int cycle) const
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
    data_out.add_data_vector (solution, solution_names);
    data_out.build_patches (p_order+2);
    data_out.write_gmv (output);
}
/*
template<int dim>
void MaxwellProblem<dim>::mesh_info(const Triangulation<dim> &tria,
               const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_indicator()]++;
          }
      }
    std::cout << " boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
         it!=boundary_count.end();
         ++it)
      {
        std::cout << it->first << "(" << it->second << " times) ";
      }
    std::cout << std::endl;
  }
  std::ofstream out (filename.c_str());
  GridOut grid_out;
  grid_out.write_eps (tria, out);
  std::cout << " written to " << filename
            << std::endl
            << std::endl;
}
 */

template <int dim>
void MaxwellProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<1; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            /*
	    GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (1);
            */
            
            GridIn<dim> gridin;
            gridin.attach_triangulation(triangulation);
            std::ifstream mesh_file("cubeincube_nobc.ucd");
            gridin.read_ucd(mesh_file);
            // print each cell and associated ids
            for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
            {
                //if (cell->at_boundary())
                //{
                    std::cout  << cell << " material: " <<  (unsigned int)cell->material_id() << " subdomain: " <<  (unsigned int)cell->
                    subdomain_id() << std::endl;
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    {
                        std::cout << f << ": " << (cell->face(f)->at_boundary()) << " " <<
                        (unsigned int)cell->face(f)->boundary_indicator() << std::endl;
                    }
                //}
            }
            //read_in_mesh("/home/ross/Dropbox/Swansea_work/meshes/sphere_in_box_1.msh");
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
int main ()
{
    try
    {
        deallog.depth_console (0);
        MaxwellProblem<3> maxwell(1);
        maxwell.run ();
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
