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

//#include <boost/math/special_functions/bessel.hpp> // Req'd for bessel functions in L-shape solution w/ singularity.

#include <fstream>
#include <iostream>


using namespace dealii;

// RIGHT HAND SIDE CLASS
// Not Currently used.
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
};
// MEMBERS
template <int dim>
RightHandSide<dim>::RightHandSide () :
Function<dim> (dim+dim)
{}
template <int dim>
inline
void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
{
    Assert (values.size() == dim+dim, ExcDimensionMismatch (values.size(), dim+dim));
    Assert (dim == 3, ExcNotImplemented());
    
    // 3D
    values(0) = 0.;
    values(1) = 0.;
    values(2) = 0.;
    values(3) = 0.;
    values(4) = 0.;
    values(5) = 0.;
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
// END RIGHT HAND SIDE CLASS


// EddyCurrent class.
template<int dim>
class EddyCurrent : public Function<dim>
{
public:
    EddyCurrent(); //const FullMatrix<double> polTensor_in_re(dim), const FullMatrix<double> &polTensor_in_im);

    virtual double value (const Point<dim> &p,
                          const unsigned int component) const;
    virtual void vector_value (const Point<dim> &p,
                               Vector<double> &result) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &values) const;
    Point<dim> centre_of_sphere;
    Vector<double> H0_re;
    Vector<double> H0_im;
private:
    const double constant_PI = numbers::PI;
    Vector<double> H_alpha;
    FullMatrix<double> polarizationTensor_re;
    FullMatrix<double> polarizationTensor_im;

    double dotprod(const Vector<double> &A, const Vector<double> &B) const;
    double dotprod(const Vector<double> &A, const Point<dim> &B) const;
};

template<int dim>
EddyCurrent<dim>::EddyCurrent() //const FullMatrix<double> &polTensor_in_re, const FullMatrix<double> &polTensor_in_im)
//(Vector<double> k_in, Vector<double> p_in)
:
Function<dim> (dim+dim),
H0_re(dim),
H0_im(dim),
polarizationTensor_re(dim),
polarizationTensor_im(dim)
{
    // Define polarization Tensor:
    //real:
    polarizationTensor_re=0;
    polarizationTensor_re(0,0)=-49.6125;//1.7078e-06;
    polarizationTensor_re(1,1)=-49.6125;//1.7078e-06;
    polarizationTensor_re(2,2)=-49.6125;//1.7078e-06;
    //imaginary:
    polarizationTensor_im=0;
    polarizationTensor_im(0,0)=0.647373;//6.8041e-07;
    polarizationTensor_im(1,1)=0.647373;//6.8041e-07;
    polarizationTensor_im(2,2)=0.647373;//6.8041e-07;
    
    //Define background field, H0:
    //real:
    H0_re(0) = 0.;
    H0_re(1) = 0.;
    H0_re(2) = 1.;
    //imaginary:
    H0_im(0) = 0.;
    H0_im(1) = 0.;
    H0_im(2) = 0.;

    centre_of_sphere(0)=0.0;
    centre_of_sphere(1)=0.0;
    centre_of_sphere(2)=0.0;
}
// EddyCurrent members:
template<int dim>
double EddyCurrent<dim>::dotprod(const Vector<double> &A, const Vector<double> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}
template<int dim>
double EddyCurrent<dim>::dotprod(const Vector<double> &A, const Point<dim> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}

template<int dim>
double EddyCurrent<dim>::value(const Point<dim> &p,
                                  unsigned int component) const
{
    Assert (dim == 3, ExcNotImplemented());
    AssertIndexRange(component, dim);
    
    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    Vector<double> w_temp_re(dim);
    Vector<double> w_temp_im(dim);
    Vector<double> v_temp(dim);
    double val;
    double rad = p.distance(centre_of_sphere);
    Vector<double> temp(dim);
    for (unsigned int i=0;i<dim;i++)
    {
        temp(i)=p(i)/rad;
    }
    rhat.outer_product(temp,temp);
    D2G=0;
    D2G.add(3.0,rhat,-1.0,eye);
    D2G*=1.0/(4.0*constant_PI*rad*rad*rad);
     
    D2G.vmult(w_temp_re, H0_re);
    D2G.vmult(w_temp_im, H0_im);
    if (component < dim)
    {        
        polarizationTensor_re.vmult(v_temp,w_temp_re);
	w_temp_im *= -1.0;
	polarizationTensor_im.vmult_add(v_temp,w_temp_im);
	
        val = v_temp(component) + H0_re(component);
    }
    else
    {
        polarizationTensor_re.vmult(v_temp,w_temp_im);
	polarizationTensor_im.vmult(v_temp,w_temp_re);
        val = v_temp(component - dim) + H0_im(component-dim);
    }
    return val;
}

template <int dim>
inline
void EddyCurrent<dim>::vector_value (const Point<dim> &p,
                                        Vector<double> &values) const
{
    Assert (values.size() == dim+dim, ExcDimensionMismatch (values.size(), dim+dim));
    
    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    Vector<double> w_temp_re(dim);
    Vector<double> w_temp_im(dim);
    Vector<double> v_temp_re(dim);
    Vector<double> v_temp_im(dim);
    
    double rad = p.distance(centre_of_sphere);
    Vector<double> temp(dim);
    for (unsigned int i=0;i<dim;i++)
    {
        temp(i)=p(i)/rad;
    }
    rhat.outer_product(temp,temp);
    D2G.add(3.0,rhat,-1.0,eye);
    D2G*=1.0/(4.0*constant_PI*rad*rad*rad);
    
    D2G.vmult(w_temp_re, H0_re);
    D2G.vmult(w_temp_im, H0_im);
    polarizationTensor_re.vmult(v_temp_re, w_temp_re);	
    polarizationTensor_re.vmult(v_temp_im, w_temp_im);
    
    w_temp_im *= -1.0;
    polarizationTensor_im.vmult_add(v_temp_re, w_temp_im);
    polarizationTensor_im.vmult_add(v_temp_im, w_temp_re);
    for (unsigned int i=0;i<dim;i++)
    {
        values(i)=v_temp_re(i) + H0_re(i);
        values(i+dim)=v_temp_im(i) + H0_im(i);
    }   
}

template <int dim>
void EddyCurrent<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double> &values,
                                      const unsigned int component) const
{
    Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    AssertIndexRange(component, dim);
    
    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    Vector<double> w_temp_re(dim);
    Vector<double> w_temp_im(dim);
    Vector<double> v_temp_re(dim);
    Vector<double> v_temp_im(dim);
    
    for (unsigned int k=0; k<points.size(); ++k) {
      const Point<dim> &p = points[k];

      double val;
      double rad = p.distance(centre_of_sphere);
      Vector<double> temp(dim);
      for (unsigned int i=0;i<dim;i++)
      {
	temp(i)=p(i)/rad;
      }
        rhat.outer_product(temp,temp);
        D2G=0;
        D2G.add(3.0,rhat,-1.0,eye);
        D2G*=1.0/(4.0*constant_PI*rad*rad*rad);
	
	D2G.vmult(w_temp_re, H0_re);
	D2G.vmult(w_temp_im, H0_im);
        
        if (component < dim)
        {
            polarizationTensor_re.vmult(v_temp_re, w_temp_re);
	    w_temp_im *= -1.0;;
	    polarizationTensor_im.vmult_add(v_temp_re, w_temp_im);
	    
            values[k] = v_temp_re(component) + H0_re(component);
        }
        else
        {
            polarizationTensor_im.vmult(v_temp_im, w_temp_re);
	    polarizationTensor_re.vmult(v_temp_im, w_temp_im);
	    
            values[k] = v_temp_im(component-dim) + H0_im(component - dim);
        }
    }
}

template <int dim>
void EddyCurrent<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    Vector<double> w_temp_re(dim);
    Vector<double> w_temp_im(dim);
    Vector<double> v_temp_re(dim);
    Vector<double> v_temp_im(dim);
    Vector<double> temp(dim);
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
        const Point<dim> &p = points[k];

        double rad = p.distance(centre_of_sphere);

        for (unsigned int i=0;i<dim;i++)
        {
            temp(i)=p(i)/rad;
        }
        rhat.outer_product(temp,temp);
        D2G=0;
        D2G.add(3.0,rhat,-1.0,eye);
        double factor = 1.0/(4.0*constant_PI*rad*rad*rad);
        if (numbers::is_finite(factor))
        {
            D2G*=factor;
        }
        else
        {
            for (unsigned int i=0; i<dim;i++)
            {
                std::cout << p(i) << " ";
            }
            std::cout << std::endl;
            D2G=0;
        }
	
        D2G.vmult(w_temp_re, H0_re);
	D2G.vmult(w_temp_im, H0_im);
        polarizationTensor_re.vmult(v_temp_re, w_temp_re);	
	polarizationTensor_re.vmult(v_temp_im, w_temp_im);
	
	w_temp_im *= -1.0;
	polarizationTensor_im.vmult_add(v_temp_re, w_temp_im);
	polarizationTensor_im.vmult_add(v_temp_im, w_temp_re);
        for (unsigned int i=0;i<dim;i++)
        {
            // Real
            value_list[k](i)=v_temp_re(i) + H0_re(i);
            // Imaginary
            value_list[k](i+dim)=v_temp_im(i) + H0_im(i);
        }
    }
}
// END WAVE PROPAGATION





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
    //void read_in_mesh (std::string mesh_file);
    void setup_system ();
    void assemble_system ();
    void solve ();
    void process_solution(const unsigned int cycle);
    void output_results_eps (const unsigned int cycle) const;
    void output_results_vtk (const unsigned int cycle) const;
    void output_results_gmv(const unsigned int cycle) const;
    void my_cross_product(Vector<double> dst, Vector<double> src1, Vector<double> src2);
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
    
    ConvergenceTable	   convergence_table;
    
    // Choose exact solution - used for BCs and error calculation
    const EddyCurrent<dim> exact_solution;

    // Input paramters (for hp-FE)
    unsigned int p_order;
    unsigned int quad_order;
    
    // Constants:
    const double constant_PI = numbers::PI;
    const double constant_epsilon0 = 0.0; // disabled for now: 8.85418782e-12; // electric constant (permittivity)
    const double constant_mu0 = 1.25663706e-6; // magnetic constant (permeability)
    const double constant_sigma0 = 1e-5; // background conductivity, set to regularise system.
    // Material Parameters:
    // Will need to be read in via input file later.
    double param_omega = 133.5; // angular frequency, rads/sec.
    double param_mu_star = 1.5*constant_mu0; // permability in conducting region.
    double param_epsilon_star = 0.0; // permit
    double param_sigma_star = 5.96e7;
    Vector<double> param_mur;
    Vector<double> param_sigma;
    Vector<double> param_epsilon;
    Vector<double> param_kappa_re; // Kappa = Kappa_re + i*Kappa_im = -omega.^2*epr + i*omega*sigma
    Vector<double> param_kappa_im;
};

template <int dim>
MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
:
dof_handler (triangulation),
// Defined as FESystem, and we need 2 FE_Nedelec - first (0) is real part, second (1) is imaginary part.
// Then need to use system blocks to solve for them.
fe (FE_Nedelec<dim>(order), 1, FE_Nedelec<dim>(order), 1),
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

// template<int dim>
// void my_cross_product(Vector<double> dst, Vector<double> src1, Vector<double> src2)
// {
//   Assert (dim==3, ExcInternalError());
//   dst(0) = src1(1)*src2(2) - src1(2)*src2(1);
//   dst(1) = src1(2)*src2(0) - src1(0)*src2(2);
//   dst(2) = src1(0)*src2(1) - src1(1)*src2(0);  
// }

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
    DoFRenumbering::block_wise (dof_handler);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
       
    // FE_Nedelec boundary condition.
    // Real part (begins at 0):
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0, exact_solution, 1, constraints);
    // Imaginary part (begins at dim):
    VectorTools::project_boundary_values_curl_conforming(dof_handler, dim, exact_solution, 1, constraints);
    
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
    param_mur.reinit(2);
    param_mur(0) = constant_mu0;
    param_mur(1) = param_mu_star;//param_mu_star/constant_mu0;
    param_sigma.reinit(2);
    param_sigma(0) = constant_sigma0;
    param_sigma(1) = param_sigma_star;

    
    //  kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    param_kappa_re.reinit(2);
    param_kappa_re(0) = -param_omega*param_omega*constant_epsilon0;
    param_kappa_re(1) = -param_omega*param_omega*param_epsilon_star;
    param_kappa_im.reinit(2);
    param_kappa_im(0) = param_omega*param_sigma(0);
    param_kappa_im(1) = param_omega*param_sigma(1);
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
    
    //FEValuesViews::Vector<dim> fe_views_re(fe_values, 0);
    //FEValuesViews::Vector<dim> fe_views_im(fe_values, dim);
    
    // Extractors to distringuish real and imaginary parts
    /* Not used right now */
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);
    
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
//    FullMatrix<double> cell_matrix_curl (dofs_per_cell, dofs_per_cell);
//    FullMatrix<double> cell_matrix_mass (dofs_per_cell, dofs_per_cell);
    double curl_part;
    double mass_part;
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    ZeroFunction<dim>      right_hand_side(fe.n_components());
    
    std::vector<Vector<double> > rhs_values (n_q_points, Vector<double>(fe.n_components()));
    std::vector<Vector<double> > neumann_value_list(n_face_q_points, Vector<double>(fe.n_components()));
    Tensor<1,dim> neumann_value_list_re(dim);
    Tensor<1,dim> neumann_value_list_im(dim);
    Tensor<1,dim> neumann_value_re(dim);
    Tensor<1,dim> neumann_value_im(dim);
    Tensor<1,dim> normal_vector;
    
    
    double current_mur;
    double current_kappa_re;
    double current_kappa_im;
    unsigned int block_index_i;
    unsigned int block_index_j;
    
    Vector<double> value(fe.n_components());
    Point<dim> temp_point;
    temp_point(0)=2.5;
    temp_point(1)=2.5;
    temp_point(2)=2.5;
    exact_solution.vector_value(temp_point,value);
    for (unsigned int i=0;i<fe.n_components();i++)
    {
      std::cout << value(i) << " ";
    }
    std::cout << std::endl;
    
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        
        current_mur = param_mur(cell->material_id()-1);//param_mur(1);//
        current_kappa_re = param_kappa_re(cell->material_id()-1);//param_kappa_re(1);//
        current_kappa_im = param_kappa_im(cell->material_id()-1);//param_kappa_im(1);//
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit (cell);
        /*
        right_hand_side_re.vector_value_list (fe_values.get_quadrature_points(),
                                              rhs_values_re);
        right_hand_side_im.vector_value_list (fe_values.get_quadrature_points(),
                                              rhs_values_im);*/
         
         exact_solution.vector_value_list (fe_values.get_quadrature_points(),
                                           rhs_values);
        

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            /*
	        std::cout << "Point: " << check_point[q_point](0) << " " << check_point[q_point](1) << " " << check_point[q_point](2) << " Vals:"
            <<rhs_values[q_point](0) << " " << rhs_values[q_point](1) << " " << rhs_values[q_point](2) << " "
            << rhs_values[q_point](3) << " " << rhs_values[q_point](4)  << " " << rhs_values[q_point](5) << std::endl;
             */
	    
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                block_index_i = fe.system_to_block_index(i).first;
                /* Write out block/base types
                std::cout << "sys2base ";
                std::cout << fe.system_to_base_index(i).first.first << " ";
                std::cout << fe.system_to_base_index(i).first.second << " ";
                std::cout << fe.system_to_base_index(i).second << std::endl;
                std::cout << "sys2block ";
                std::cout << fe.system_to_block_index(i).first << " ";
                std::cout << fe.system_to_block_index(i).second << std::endl;
                */
		
                /*
                value_i[0] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+0);
                value_i[1] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+1);
                value_i[2] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+2);
                */
		/*
                value_i = fe_values[E_re].value(i,q_point) + fe_values[E_im].value(i,q_point);
		std::cout << "Re: " << fe_values[E_re].value(i,q_point)[0] << " "
		<< fe_values[E_re].value(i,q_point)[1] << " "
		<< fe_values[E_re].value(i,q_point)[2] << std::endl;
		std::cout << "Im: " << fe_values[E_im].value(i,q_point)[0] << " "
		<< fe_values[E_im].value(i,q_point)[1] << " "
		<< fe_values[E_im].value(i,q_point)[2] << std::endl;
		*/
                		
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    block_index_j = fe.system_to_block_index(j).first;
                    /*
                    value_j[0] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+0);
                    value_j[1] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+1);
                    value_j[2] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+2);
                     */
                    // Note that since real and imaginary parts use same basis, then no need to compute different types.
                    // Could simply compute for a single block then re-use to construct the full 2x2 block system.
                    /*curl_part = (dotprod(fe_views_re.curl(i,q_point),fe_views_re.curl(j,q_point))
				+ dotprod(fe_views_im.curl(i,q_point),fe_views_im.curl(j,q_point)))*fe_values.JxW(q_point);
                    mass_part = dotprod(value_i,value_j)*fe_values.JxW(q_point);
                    */
                    
                    if (block_index_i == block_index_j)
                    //    cell_matrix(i,j) += (1.0/current_mur)*curl_part + current_kappa_re*mass_part;
                        cell_matrix(i,j) += ((1.0/current_mur)*(fe_values[E_re].curl(i,q_point)*fe_values[E_re].curl(j,q_point)
                                                              + fe_values[E_im].curl(i,q_point)*fe_values[E_im].curl(j,q_point))
                                              + current_kappa_re*(fe_values[E_re].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                                                + fe_values[E_im].value(i,q_point)*fe_values[E_im].value(j,q_point))
                                             )*fe_values.JxW(q_point);
                    else
                        if (block_index_i == 0)
                      //      cell_matrix(i,j) += -current_kappa_im*mass_part;
                            cell_matrix(i,j) += -current_kappa_im*(   fe_values[E_re].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                                                    + fe_values[E_im].value(i,q_point)*fe_values[E_im].value(j,q_point)
                                                                   )*fe_values.JxW(q_point);
                        else if (block_index_i == 1)
                            cell_matrix(i,j) += current_kappa_im*(  fe_values[E_re].value(i,q_point)*fe_values[E_re].value(j,q_point)
                                                                  + fe_values[E_im].value(i,q_point)*fe_values[E_im].value(j,q_point)
                                                                  )*fe_values.JxW(q_point);
                        //    cell_matrix(i,j) += current_kappa_im*mass_part;
                    

                }
                /*
                if (block_index_i == 0)
                    cell_rhs(i) += (fe_values[E_re].value(i,q_point)*rhs_values_re[q_point])*fe_values.JxW(q_point);
                    //cell_rhs(i) += dotprod(value_i,rhs_values_re[q_point])*fe_values.JxW(q_point);
                else if (block_index_i == 1)
                    cell_rhs(i) += (fe_values[E_im].value(i,q_point)*rhs_values_re[q_point])*fe_values.JxW(q_point);
                    //cell_rhs(i) += dotprod(value_i,rhs_values_im[q_point])*fe_values.JxW(q_point);
                 */
                
                
            }
        }
        // Loop over faces for neumann condition:
        for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if (cell->face(face_number)->at_boundary()
                &&
                (cell->face(face_number)->boundary_indicator() == 0)) // Will need to be updated to 1 once mesh files match this.
            {
                fe_face_values.reinit (cell, face_number);
                exact_solution.vector_value_list(fe_face_values.get_quadrature_points(),neumann_value_list);
                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                {
                    for (unsigned int component=0; component<dim; component++)
                    {
                        neumann_value_list_re[component] = (1.0/current_mur)*exact_solution.H0_re(component);//(1.0/urrent_mur)*neumann_value_list[q_point](component);
                        neumann_value_list_im[component] = 0;//(1.0/current_mur)*neumann_value_list[q_point](component+dim);
                        normal_vector[component] = fe_face_values.normal_vector(q_point)(component);
                    }
                    cross_product(neumann_value_re, normal_vector, neumann_value_list_re);
                    cross_product(neumann_value_im, normal_vector, neumann_value_list_im);
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        cell_rhs(i) += -(neumann_value_re*fe_face_values[E_re].value(i,q_point)*fe_face_values.JxW(q_point));
                        cell_rhs(i) += -(neumann_value_im*fe_face_values[E_im].value(i,q_point)*fe_face_values.JxW(q_point));
                    }
                }
            }
        }
        /* Output local matrix
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                std::cout << cell_matrix(i,j) << " ";
            }
            std::cout << ";" << std::endl;
        }
        */
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
//    const WavePropagation_Re<dim> exact_solution_re;
//    const WavePropagation_Im<dim> exact_solution_im;
    
    // Masks for real & imaginary parts
    const ComponentSelectFunction<dim> E_re_mask (std::make_pair(0,dim), dim+dim);
    const ComponentSelectFunction<dim> E_im_mask (std::make_pair(dim, dim+dim), dim+dim);
    
    Vector<double> diff_per_cell_re(triangulation.n_active_cells());
    Vector<double> diff_per_cell_im(triangulation.n_active_cells());
    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell_re, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm,
                                      &E_re_mask);
    const double L2_error_re = diff_per_cell_re.l2_norm();
    
    VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                      diff_per_cell_im, QGauss<dim>(quad_order+1),
                                      VectorTools::L2_norm,
                                      &E_im_mask);
    const double L2_error_im = diff_per_cell_im.l2_norm();
    /*
    std::cout << "sol: " << std::endl;
    for (unsigned int i=0; i<solution.size();i++)
    {
      std::cout << solution(i) << std::endl;
    }
    
    std::cout << "diffs: " << std::endl;
    for (unsigned int i=0;i<triangulation.n_active_cells();i++)
    {
        std::cout << diff_per_cell_re(i) << " " << diff_per_cell_im(i) << std::endl;
    }
    */
    
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("E_re L2 Error", L2_error_re);
    convergence_table.add_value("E_im L2 Error", L2_error_im);
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
    data_out.build_patches (p_order);
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
            // Cube mesh
//             GridGenerator::hyper_cube (triangulation, -0.5, 0.5);
//             triangulation.refine_global (2);


            /* Read in mesh: */
            GridIn<dim> gridin;
            gridin.attach_triangulation(triangulation);
            std::ifstream mesh_file("../meshes/sphereincube_nobc.ucd");
            gridin.read_ucd(mesh_file);
            
                            
                std::cout << exact_solution.centre_of_sphere(0) << " "
                << exact_solution.centre_of_sphere(1) << " "
                << exact_solution.centre_of_sphere(2) << std::endl;
	    
            /* print each cell and associated ids */
            for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
            {
// print cell containing centre of sphere:
                if (cell->point_inside(exact_solution.centre_of_sphere))
                {
                    std::cout  << cell << " material: " <<  (unsigned int)cell->material_id() << " subdomain: " <<  (unsigned int)cell->
                    subdomain_id() << std::endl;
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    {
                        std::cout << f << ": " << (cell->face(f)->at_boundary()) << " " <<
                        (unsigned int)cell->face(f)->boundary_indicator();
                        std::cout << std::endl;
                        for (unsigned int verti=0;verti<4;verti++)
                        {
                            std::cout << verti << " " << cell->face(f)->vertex(verti)(0) << " "
                            << cell->face(f)->vertex(verti)(1) << " "
                            << cell->face(f)->vertex(verti)(2);
                            std::cout << std::endl;
                        }
                    }
                }
                
                /* print all material/boundary info:
                if (cell->at_boundary())
                {
                    std::cout  << cell << " material: " <<  (unsigned int)cell->material_id() << " subdomain: " <<  (unsigned int)cell->
                    subdomain_id() << std::endl;
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                    {
                        std::cout << f << ": " << (cell->face(f)->at_boundary()) << " " <<
                        (unsigned int)cell->face(f)->boundary_indicator();
                        std::cout << std::endl;
                        for (unsigned int verti=0;verti<4;verti++)
                        {
                            std::cout << verti << " " << cell->face(f)->vertex(verti)(0) << " "
                            << cell->face(f)->vertex(verti)(1) << " "
                            << cell->face(f)->vertex(verti)(2);
                            std::cout << std::endl;
                        }
                    }
                }
                 */
            }
            //read_in_mesh("/home/ross/Dropbox/Swansea_work/meshes/sphere_in_box_1.msh");
            //triangulation.refine_global(2);
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
    convergence_table.set_precision("E_re L2 Error",8);
    convergence_table.set_scientific("E_re L2 Error",true);
    
    convergence_table.set_precision("E_im L2 Error",8);
    convergence_table.set_scientific("E_im L2 Error",true);
    
    convergence_table.write_text(std::cout);
}
// END MAXWELL CLASS
int main ()
{
    try
    {
        deallog.depth_console (0);
        MaxwellProblem<3> maxwell(0);
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
