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
#include <deal.II/numerics/data_postprocessor.h>

//#include <boost/math/special_functions/bessel.hpp> // Req'd for bessel functions in L-shape solution w/ singularity.

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <locale>
#include <string>


using namespace dealii;


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
    double sphere_radius;
    Vector<double> H0_re;
    Vector<double> H0_im;
private:
    const double constant_PI = numbers::PI;
    FullMatrix<double> polarizationTensor_re;
    FullMatrix<double> polarizationTensor_im;

    double dotprod(const Vector<double> &A, const Vector<double> &B) const;
    double dotprod(const Vector<double> &A, const Point<dim> &B) const;
};

template<int dim>
EddyCurrent<dim>::EddyCurrent()
:
Function<dim> (dim+dim),
H0_re(dim),
H0_im(dim),
polarizationTensor_re(dim),
polarizationTensor_im(dim)
{
    // Define polarization Tensor:
    // sphere radius 0.1: -0.0046626+0.0013633i
    // as above, sigma=5.96e5:  0.0017078+0.00068041i
    //real:
    polarizationTensor_re=0.0;
    polarizationTensor_re(0,0)=0.0017078;
    polarizationTensor_re(1,1)=0.0017078;
    polarizationTensor_re(2,2)=0.0017078;
    //imaginary:
    polarizationTensor_im=0.0;
    polarizationTensor_im(0,0)=0.00068041;
    polarizationTensor_im(1,1)=0.00068041;
    polarizationTensor_im(2,2)=0.00068041;
    
    //Define background field, H0:
    //real:
    H0_re(0) = 0.0;
    H0_re(1) = 0.0;
    H0_re(2) = 1.0;
    //imaginary:
    H0_im(0) = 0.0;
    H0_im(1) = 0.0;
    H0_im(2) = 0.0;

    centre_of_sphere(0)=0.0;
    centre_of_sphere(1)=0.0;
    centre_of_sphere(2)=0.0;
    sphere_radius=0.1;
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
        val = -( v_temp(component - dim) + H0_im(component-dim) ); // minus as the solution is the complex conjugate of what we need.
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
        values(i+dim)= - ( v_temp_im(i) + H0_im(i) );// minus as the solution is the complex conjugate of what we need.
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
	    
            values[k] = -( v_temp_im(component-dim) + H0_im(component - dim) ); // minus as the solution is the complex conjugate of what we need.
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
            value_list[k](i+dim)= -( v_temp_im(i) + H0_im(i) ); // minus as the solution is the complex conjugate of what we need.
        }
    }
}
// END EDDY CURRENT CLASS

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
    class ComputeH; // If setting up class within this class for DataPostprocessor.
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
    const double constant_sigma0 = 0.1; // background conductivity, set to regularise system (can't solve curlcurlE = f).
    // Material Parameters:
    // Will need to be read in via input file later.
    double param_omega = 133.5; // angular frequency, rads/sec.
    double param_mu_star = 1.0; // permability in conducting region.
    double param_epsilon_star = 0.0; // permitivity in conducting region.
    double param_sigma_star = 0.1;//5.96e5; // conductivity in conducting region
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
    quad_order = p_order+3;
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
    param_mur.reinit(2);
    param_mur(0) = 1;
    param_mur(1) = param_mu_star;
    param_sigma.reinit(param_mur.size());
    param_sigma(0) = constant_sigma0*constant_mu0;
    param_sigma(1) = param_sigma_star*constant_mu0;
    param_epsilon.reinit(param_mur.size());
    param_epsilon(0) = constant_epsilon0*constant_mu0;
    param_epsilon(1) = param_epsilon_star*constant_mu0;

    
    //  kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    param_kappa_re.reinit(param_mur.size());
    param_kappa_im.reinit(param_mur.size());
    for (unsigned int i=0;i<param_mur.size();i++) // note mur and kappa must have same size:
    {
        param_kappa_re(i) = -param_omega*param_omega*param_epsilon(i);
        param_kappa_im(i) = param_omega*param_sigma(i);
    }
    
    //Geometric parts:
    centre_of_sphere = exact_solution.centre_of_sphere;
    sphere_radius = exact_solution.sphere_radius;
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
    
    /* Quick check on exact solution:     */
    Vector<double> value(fe.n_components());
    Point<dim> temp_point;
    temp_point(0)=0.823;
    temp_point(1)=0.924;
    temp_point(2)=0.824;
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
        fe_values.reinit (cell);
        current_mur = param_mur(cell->material_id());
        current_kappa_re = param_kappa_re(cell->material_id());
        current_kappa_im = param_kappa_im(cell->material_id());
        cell_matrix = 0;
        cell_rhs = 0;

        // Loop over quad points:
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                block_index_i = fe.system_to_block_index(i).first;
//                 if (cell == dof_handler.begin_active())
//                 {
//                     std::cout << i << ", " << block_index_i << ": " <<  fe_values[E_re].value(i,q_point) << " "
//                     << fe_values[E_im].value(i,q_point) << std::endl;
//                 }
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
//                         cell_matrix(i,j) += pow(-1.0,block_index_i+1)*current_kappa_im*(  fe_values[E_im].value(i,q_point)*fe_values[E_re].value(j,q_point)
//                                                                 + fe_values[E_re].value(i,q_point)*fe_values[E_im].value(j,q_point)
//                                                                    )*fe_values.JxW(q_point);
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
                (cell->face(face_number)->boundary_indicator() == 1))
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
                        neumann_value_list_re[component] = (constant_mu0/param_omega)*neumann_value_list[q_point](component+dim);
                        neumann_value_list_im[component] = -(constant_mu0/param_omega)*neumann_value_list[q_point](component);
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
    
        /* Quick check on exact solution:     */
    Vector<double> value(fe.n_components());
    Point<dim> temp_point;
    temp_point(0)=0.123;
    temp_point(1)=0.324;
    temp_point(2)=0.124;
    VectorTools::point_value (dof_handler,solution, temp_point, value);
    for (unsigned int i=0;i<fe.n_components();i++)
    {
      std::cout << value(i) << " ";
    }
    std::cout << std::endl;
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
private:
    Point<dim> centre_of_sphere;
    double sphere_radius = 0.1;
    
    // Constants, MUST REPLACE WITH EQNDATA LATER:
    const double constant_PI = numbers::PI;
    const double constant_epsilon0 = 0.0; // disabled for now: 8.85418782e-12; // electric constant (permittivity)
    const double constant_mu0 = 1.25663706e-6; // magnetic constant (permeability)
    const double constant_sigma0 = 0.1; // background conductivity, set to regularise system (can't solve curlcurlE = f).
    // Material Parameters:
    // Will need to be read in via input file later.
    double param_omega = 133.5; // angular frequency, rads/sec.
    double param_mu_star = 1.0; // permability in conducting region.
    double param_epsilon_star = 0.0; // permitivity in conducting region.
    double param_sigma_star = 0.1;//5.96e5; // conductivity in conducting region
    Vector<double> param_mur;
    Vector<double> param_sigma;
    Vector<double> param_epsilon;
    Vector<double> param_kappa_re; // Kappa = Kappa_re + i*Kappa_im = -omega.^2*epr + i*omega*sigma
    Vector<double> param_kappa_im;
};

template <int dim>
MaxwellProblem<dim>::ComputeH::ComputeH ()
:
DataPostprocessor<dim>()
{
    centre_of_sphere(0)=0;
    centre_of_sphere(1)=0.0;
    centre_of_sphere(2)=0.0;
    // Setup material parameters:
    param_mur.reinit(2);
    param_mur(0) = 1.0;
    param_mur(1) = 1.0;//param_mu_star/constant_mu0;
    
    //  kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    param_kappa_re.reinit(1);
    param_kappa_re(0) =0.0;
    param_kappa_re(1) = 0.0;
    param_kappa_im.reinit(2);
    param_kappa_im(0) = param_omega*constant_sigma0;
    param_kappa_im(1) = param_omega*param_sigma_star;
}

template <int dim>
std::vector<std::string>
MaxwellProblem<dim>::ComputeH::get_names() const
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
        double rad = centre_of_sphere.distance(evaluation_points[q]);
        if (rad <= sphere_radius)
        {
            temp_scaling = param_omega/(param_mur(1)*constant_mu0);
        }
        else
        {
            temp_scaling = param_omega/(param_mur(0)*constant_mu0);
            
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

        //wrong, delete when sure:
//        // real part:
//        computed_quantities[q](0+2*dim) = (duh[q][3][1]-duh[q][0][2])*temp_scaling;
//        computed_quantities[q](1+2*dim) = (duh[q][4][2]-duh[q][1][0])*temp_scaling;
//        computed_quantities[q](2+2*dim) = (duh[q][5][0]-duh[q][2][1])*temp_scaling;
//        //imaginary part
//        computed_quantities[q](0+3*dim) = -(duh[q][0][1]-duh[q][3][2])*temp_scaling;
//        computed_quantities[q](1+3*dim) = -(duh[q][1][2]-duh[q][4][0])*temp_scaling;
//        computed_quantities[q](2+3*dim) = -(duh[q][2][0]-duh[q][5][1])*temp_scaling;
        
    }
}
// END Compute magnetic field class

//Better way:
//
// template <int dim>
// class MaxwellProblem<dim>::Postprocessor : public DataPostprocessor<dim>
// {
// public:
//    Postprocessor() : DataPostprocessor<dim>() {}
//    virtual void
//    compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
//                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
//                                       const std::vector<std::vector<Tensor<2,dim> > > &dduh,
//                                       const std::vector<Point<dim> >                  &normals,
//                                       const std::vector<Point<dim> >                  &evaluation_points,
//                                       std::vector<Vector<double> >                    &computed_quantities) const;
//    virtual std::vector<std::string> get_names () const;
//    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>   get_data_component_interpretation () const;
//    virtual UpdateFlags get_needed_update_flags () const;
// };
// 
// // template <int dim>
// // MaxwellProblem<dim>::Postprocessor::
// //Postprocessor()
// // :
// // {}
// 
// template <int dim>
// std::vector<std::string>
// MaxwellProblem<dim>::Postprocessor::get_names() const
// {
//    std::vector<std::string> solution_names (dim, "E_re");
//    solution_names.push_back ("E_im");
//    solution_names.push_back ("E_im");
//    solution_names.push_back ("E_im");
// //    solution_names.push_back ("H_re");
// //    solution_names.push_back ("H_re");
// //    solution_names.push_back ("H_re");
// //    solution_names.push_back ("H_im");
// //    solution_names.push_back ("H_im");
// //    solution_names.push_back ("H_im");
// 
//    return solution_names;
// }
// template <int dim>
// std::vector<DataComponentInterpretation::DataComponentInterpretation>
// MaxwellProblem<dim>::Postprocessor::
// get_data_component_interpretation () const
// {
//    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//    interpretation (dim,
//                    DataComponentInterpretation::component_is_part_of_vector);
//    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
//    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
//    interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
//    return interpretation;
// }
// template <int dim>
// UpdateFlags
// MaxwellProblem<dim>::Postprocessor::get_needed_update_flags() const
// {
//    return update_values | update_gradients | update_q_points;
// }
// template <int dim>
// void
// MaxwellProblem<dim>::Postprocessor::
// compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
//                                   const std::vector<std::vector<Tensor<1,dim> > > &duh,
//                                   const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
//                                   const std::vector<Point<dim> >                  &/*normals*/,
//                                   const std::vector<Point<dim> >                  &/*evaluation_points*/,
//                                   std::vector<Vector<double> >                    &computed_quantities) const
// {
//    const unsigned int n_quadrature_points = uh.size();
//    Assert (duh.size() == n_quadrature_points,                  ExcInternalError());
//    Assert (computed_quantities.size() == n_quadrature_points,  ExcInternalError());
//    Assert (uh[0].size() == dim+dim,                              ExcInternalError());
//    for (unsigned int q=0; q<n_quadrature_points; ++q)
//    {
//        for (unsigned int d=0; d<dim+dim; ++d)
//        {
//            computed_quantities[q](d) = uh[q](d);
//            
//            
// //            computed_quantities[q](d)
// //            = (uh[q](d) *  EquationData::year_in_seconds * 100);
// //        const double pressure = (uh[q](dim)-minimal_pressure);
// //        computed_quantities[q](dim) = pressure;
// //        const double temperature = uh[q](dim+1);
// //        computed_quantities[q](dim+1) = temperature;
// //        Tensor<2,dim> grad_u;
// //        for (unsigned int d=0; d<dim; ++d)
// //            grad_u[d] = duh[q][d];
// //        const SymmetricTensor<2,dim> strain_rate = symmetrize (grad_u);
// //        computed_quantities[q](dim+2) = 2 * EquationData::eta *
// //        strain_rate * strain_rate;
// //        computed_quantities[q](dim+3) = partition;
//         }
//    }
// }
// END CLEVER WAY.

//Naive way:
// template <int dim>
// class ComputeH : public DataPostprocessorVector<dim>
// {
// public:
//    ComputeH ();
//    virtual
//    void
//    compute_derived_quantities_vector (const std::vector< Vector< double > > &uh,
//                                       const std::vector< std::vector< Tensor< 1, dim > > > &duh,
//                                       const std::vector< std::vector< Tensor< 2, dim > > > &dduh,
//                                       const std::vector< Point< dim > > &normals,
//                                       const std::vector<Point<dim> > &evaluation_points,
//                                       std::vector< Vector< double > > &computed_quantities) const;
// };
// template <int dim>
// ComputeH<dim>::ComputeH ()
// :
// DataPostprocessorScalar<dim> ("Intensity",
//                              update_values)
// {}
// template <int dim>
// void
// ComputeH<dim>::compute_derived_quantities_vector (
//                                                          const std::vector< Vector< double > >                  &uh,
//                                                          const std::vector< std::vector< Tensor< 1, dim > > >  & /*duh*/,
//                                                          const std::vector< std::vector< Tensor< 2, dim > > >  & /*dduh*/,
//                                                          const std::vector< Point< dim > >                     & /*normals*/,
//                                                          const std::vector<Point<dim> >                        & /*evaluation_points*/,
//                                                          std::vector< Vector< double > >                        &computed_quantities
//                                                          ) const
// {
//    Assert(computed_quantities.size() == uh.size(),
//           ExcDimensionMismatch (computed_quantities.size(), uh.size()));
//    for (unsigned int i=0; i<computed_quantities.size(); i++)
//    {
//        Assert(computed_quantities[i].size() == 1,
//               ExcDimensionMismatch (computed_quantities[i].size(), 1));
//        Assert(uh[i].size() == 2, ExcDimensionMismatch (uh[i].size(), 2));
//        computed_quantities[i](0) = std::sqrt(uh[i](0)*uh[i](0) + uh[i](1)*uh[i](1));
//    }
// }
// END Compute magnetic field class

template <int dim>
void MaxwellProblem<dim>::output_results_vtk (const unsigned int cycle) const
{
    
    std::ostringstream filename;
    filename << "solution-" << cycle << ".vtk";
    std::ofstream output (filename.str().c_str());
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    // simple way, (no DataPostprocessor):
    /*
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
                              */
    
    // DataPostprocessor way:
    ComputeH postprocessor;
    
    data_out.add_data_vector(solution,postprocessor);
    data_out.build_patches (quad_order);
    data_out.write_vtk (output);
}

template <int dim>
void MaxwellProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle<1; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            // Cube mesh
//             GridGenerator::hyper_cube (triangulation, -1, 1);
//             triangulation.refine_global (1);

            /* Read in mesh: */
            std::string mesh_name = "../meshes/sphere_in_box_nobc_try2.ucd";
            read_in_mesh(mesh_name);
            
            // Set boundaries to neumann (boundary_id = 1)
            typename Triangulation<dim>::cell_iterator
            cell = triangulation.begin (),
            endc = triangulation.end();
            for (; cell!=endc; ++cell)
            {
                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    if (cell->face(face)->at_boundary())
                    {
                        cell->face(face)->set_boundary_indicator (1);
                    }
                }          
            }
            triangulation.refine_global (1);
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
