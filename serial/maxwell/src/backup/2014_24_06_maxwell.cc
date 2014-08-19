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

#include <boost/math/special_functions/bessel.hpp> // Req'd for bessel functions in L-shape solution w/ singularity.

#include <fstream>
#include <iostream>


using namespace dealii;

// Toy Trig Solution CLASS
template<int dim>
class ToyTrigSolution : public Function<dim>
{
public:
    ToyTrigSolution () : Function<dim>(dim) {}
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
// Toy Trig Solution MEMBERS
template<int dim>
double ToyTrigSolution<dim>::value(const Point<dim> &p,
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
void ToyTrigSolution<dim>::vector_value(const Point<dim> &p,
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
void ToyTrigSolution<dim>::value_list (const std::vector<Point<dim> > &points,
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
void ToyTrigSolution<dim>::vector_value_list (const std::vector<Point<dim> > &points,
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
// RHS:
template <int dim>
class ToyTrigSolutionRHS :  public Function<dim>
{
public:
    ToyTrigSolutionRHS ();
    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
private:
    const double PI = dealii::numbers::PI;
	const double bc_constant = 0.1;
};
template <int dim>
ToyTrigSolutionRHS<dim>::ToyTrigSolutionRHS () :
Function<dim> (dim)
{}
template <int dim>
inline
void ToyTrigSolutionRHS<dim>::vector_value (const Point<dim> &p,
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
void ToyTrigSolutionRHS<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
    Assert (value_list.size() == points.size(), ExcDimensionMismatch (value_list.size(), points.size()));
    const unsigned int n_points = points.size();
    for (unsigned int p=0; p<n_points; ++p)
    {
        ToyTrigSolutionRHS<dim>::vector_value (points[p], value_list[p]);
    }
}

// END Toy Trig Solution


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
    const double PI = dealii::numbers::PI;
	const double bc_constant = 0.1;
};
// MEMBERS
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
// END RIGHT HAND SIDE CLASS


// WAVE PROPAGATION CLASS.
template<int dim>
class WavePropagation_Re : public Function<dim> 
{
public:
    WavePropagation_Re();// (Vector<double> k_in, Vector<double> p_in);

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
  Vector<double> k_wave;
  Vector<double> p_wave;
  double dotprod(const Vector<double> &A, const Vector<double> &B) const;
  double dotprod(const Vector<double> &A, const Point<dim> &B) const;  
};

template<int dim>
WavePropagation_Re<dim>::WavePropagation_Re()
//(Vector<double> k_in, Vector<double> p_in)
:
Function<dim> (dim)
{
  /* Removed input for now - need to define here:
  k_wave=k_in;
  p_wave=p_in;
  */
  k_wave.reinit(3);
  k_wave(0) = 1.;
  k_wave(1) = 1.;
  k_wave(2) = 0.;
  p_wave.reinit(3);
  p_wave(0) = 0.;
  p_wave(1) = 0.;
  p_wave(2) = 1.;
  p_wave(2) = 1.;
}
// Wave Propagation members:
template<int dim>
double WavePropagation_Re<dim>::dotprod(const Vector<double> &A, const Vector<double> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}
template<int dim>
double WavePropagation_Re<dim>::dotprod(const Vector<double> &A, const Point<dim> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}

template<int dim>
double WavePropagation_Re<dim>::value(const Point<dim> &p,
                                  unsigned int component) const
{
    Assert (dim >= 2, ExcNotImplemented());
    AssertIndexRange(component, dim);
    
    double exponent = dotprod(k_wave, p);
            
    double val = p_wave(component)*std::cos(exponent);
    return val;
    
}

template <int dim>
inline
void WavePropagation_Re<dim>::vector_value (const Point<dim> &p,
                                        Vector<double> &values) const
{
    Assert (values.size() == 3, ExcDimensionMismatch (values.size(), 3));
    
    double exponent = dotprod(k_wave, p);
    for (unsigned int k = 0; k < dim; k++){
      values(k) = p_wave(k)*std::cos(exponent);
    }
    
    
}

template <int dim>
void WavePropagation_Re<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double> &values,
                                      const unsigned int component) const
{
    Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    AssertIndexRange(component, dim);
    
    double exponent;
    for (unsigned int i=0; i<points.size(); ++i) {
      const Point<dim> &p = points[i];
      exponent = dotprod(k_wave, p);
      values[i] = p_wave(component)*std::cos(exponent);
    }

    
}

template <int dim>
void WavePropagation_Re<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();

    double exponent;
    for (unsigned int i=0; i<points.size(); ++i)
    {
        const Point<dim> &p = points[i];
	exponent = dotprod(k_wave, p);
	value_list[i](0) = p_wave(0)*std::cos(exponent);
	value_list[i](1) = p_wave(1)*std::cos(exponent);
	value_list[i](2) = p_wave(2)*std::cos(exponent);

    }
}
// Imaginary part:
// WAVE PROPAGATION CLASS.
template<int dim>
class WavePropagation_Im : public Function<dim> 
{
public:
    WavePropagation_Im ();//(Vector<double> k_in, Vector<double> p_in);

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
  Vector<double> k_wave;
  Vector<double> p_wave;
  double dotprod(const Vector<double> &A, const Vector<double> &B) const;
  double dotprod(const Vector<double> &A, const Point<dim> &B) const;  
};
template<int dim>
WavePropagation_Im<dim>::WavePropagation_Im()
//(Vector<double> k_in, Vector<double> p_in)
:
Function<dim> (dim)
{
  /* Removed input for now - need to define here:
  k_wave=k_in;
  p_wave=p_in;
  */
  k_wave.reinit(3);
  k_wave(0) = 1.;
  k_wave(1) = 1.;
  k_wave(2) = 0.;
  p_wave.reinit(3);
  p_wave(0) = 0.;
  p_wave(1) = 0.;
  p_wave(2) = 1.;
}
// Wave Propagation members:
template<int dim>
double WavePropagation_Im<dim>::dotprod(const Vector<double> &A, const Vector<double> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}
template<int dim>
double WavePropagation_Im<dim>::dotprod(const Vector<double> &A, const Point<dim> &B) const
{
    double return_val = 0;
    for(unsigned int k = 0; k < dim; k++) {
        return_val += A(k)*B(k);
    }
    return return_val;
}

template<int dim>
double WavePropagation_Im<dim>::value(const Point<dim> &p,
                                  unsigned int component) const
{
    Assert (dim >= 2, ExcNotImplemented());
    AssertIndexRange(component, dim);
    
    double exponent = dotprod(k_wave, p);
            
    double val = p_wave(component)*std::sin(exponent);
    return val;
    
}

template <int dim>
inline
void WavePropagation_Im<dim>::vector_value (const Point<dim> &p,
                                        Vector<double> &values) const
{
    Assert (values.size() == 3, ExcDimensionMismatch (values.size(), 3));
    
    double exponent = dotprod(k_wave, p);
    for (unsigned int k = 0; k < dim; k++){
      values(k) = p_wave(k)*std::sin(exponent);
    }
    
    
}

template <int dim>
void WavePropagation_Im<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double> &values,
                                      const unsigned int component) const
{
    Assert (values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    AssertIndexRange(component, dim);
    
    double exponent;
    for (unsigned int i=0; i<points.size(); ++i) {
      const Point<dim> &p = points[i];
      exponent = dotprod(k_wave, p);
      values[i] = p_wave(component)*std::sin(exponent);
    }   
}

template <int dim>
void WavePropagation_Im<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list) const
{
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    const unsigned int n_points = points.size();

    double exponent;
    for (unsigned int i=0; i<points.size(); ++i)
    {
        const Point<dim> &p = points[i];
	exponent = dotprod(k_wave, p);
        value_list[i](0) = p_wave(0)*std::sin(exponent);
        value_list[i](1) = p_wave(1)*std::sin(exponent);
	value_list[i](2) = p_wave(2)*std::sin(exponent);
    }
}
// END WAVE PROPAGATION


// BESSEL SOLUTION CLASS. USE FOR L-SHAPE WITH SINGULARITY.
template<int dim>
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
// END BESSEL SOLUTION CLASS


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
    WavePropagation_Re<dim> exact_solution_re;
    WavePropagation_Im<dim> exact_solution_im;

    // Input paramters (for hp-FE)
    unsigned int p_order;
    unsigned int quad_order;
    
    // Constants:
    const double constant_PI = numbers::PI;
    const double constant_epsilon0 = 8.85418782e-12; // electric constant (permittivity)
    const double constant_mu0 = 1.25663706e-6; // magnetic constant (permeability)
    const double constant_sigma0 = 1e-5; // background conductivity, set to regularise system.
    // Material Parameters:
    double param_omega = sqrt(2.0); // angular frequency, rads/sec. Set elsewhere later.
    double param_mu_star = 2e6; // permability in conducting region. Will be set elsewhere later.
    Vector<double> param_mur;
    Vector<double> param_kappa_re; // Kappa = Kappa_re + i*Kappa_im = -omega.^2*epr + i*omega*sigma
    Vector<double> param_kappa_im;
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
    DoFRenumbering::block_wise (dof_handler);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    // FE_Nedelec boundary condition.
    // Real part (begins at 0):
    VectorTools::project_boundary_values_curl_conforming(dof_handler, 0,
							 exact_solution_re, 0, constraints);
    // Imaginary part (begins at dim):
    VectorTools::project_boundary_values_curl_conforming(dof_handler, dim,
							 exact_solution_im, 0, constraints);
    
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
    param_mur(0) = 1.;
    param_mur(1) = 1.;//param_mu_star/constant_mu0;
    
    //  kappa = -omega^2*epr + i*omega*sigma;
    // i.e. kappa_re = -omega^2
    //      kappa_im = omega*sigma
    param_kappa_re.reinit(2);
    param_kappa_re(0) = -param_omega*param_omega;
    param_kappa_re(1) = -param_omega*param_omega;
    param_kappa_im.reinit(2);
    param_kappa_im(0) = 0.;
    param_kappa_im(1) = 0.;
}
template <int dim>
void MaxwellProblem<dim>::assemble_system ()
{
    const QGauss<dim>  quadrature_formula(quad_order);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values);
    FEValuesViews::Vector<dim> fe_views_re(fe_values, 0);
    FEValuesViews::Vector<dim> fe_views_im(fe_values, dim);
    
    // Extractors to distringuish real and imaginary parts
    /* Not used right now
    const FEValuesExtractors::Vector E_re(0);
    const FEValuesExtractors::Vector E_im(dim);*/
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
//    FullMatrix<double> cell_matrix_curl (dofs_per_cell, dofs_per_cell);
//    FullMatrix<double> cell_matrix_mass (dofs_per_cell, dofs_per_cell);
    double curl_part;
    double mass_part;
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    ZeroFunction<dim>      right_hand_side_re(dim);
    ZeroFunction<dim>      right_hand_side_im(dim);
    std::vector<Vector<double> > rhs_values_re (n_q_points,
                                             Vector<double>(dim));
    std::vector<Vector<double> > rhs_values_im (n_q_points,
                                             Vector<double>(dim));
    Tensor<1,dim> value_i, value_j;
    
    double current_mur;
    double current_kappa_re;
    double current_kappa_im;
    unsigned int block_index_i;
    unsigned int block_index_j;
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        
        current_mur = param_mur(0);//param_mur(cell->material_id()-1);
        current_kappa_re = param_kappa_re(0);//param_kappa_re(cell->material_id()-1);
        current_kappa_im = param_kappa_im(0);;//param_kappa_im(cell->material_id()-1);
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit (cell);
        right_hand_side_re.vector_value_list (fe_values.get_quadrature_points(),
                                              rhs_values_re);
        right_hand_side_im.vector_value_list (fe_values.get_quadrature_points(),
                                              rhs_values_im);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
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
		
		
		value_i[0] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+0);
                value_i[1] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+1);
                value_i[2] = fe_values.shape_value_component(i,q_point,(dim)*block_index_i+2);
                		
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    block_index_j = fe.system_to_block_index(j).first;
                    value_j[0] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+0);
                    value_j[1] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+1);
                    value_j[2] = fe_values.shape_value_component(j,q_point,(dim)*block_index_j+2);
                    // Note that since real and imaginary parts use same basis, then no need to compute different types.
                    // Could simply compute for a single block then re-use to construct the full 2x2 block system.
                    curl_part = (dotprod(fe_views_re.curl(i,q_point),fe_views_re.curl(j,q_point))
				+ dotprod(fe_views_im.curl(i,q_point),fe_views_im.curl(j,q_point)))*fe_values.JxW(q_point);
                    mass_part = dotprod(value_i,value_j)*fe_values.JxW(q_point);
                    
                    if (block_index_i == block_index_j)
                        cell_matrix(i,j) += (1.0/current_mur)*curl_part + current_kappa_re*mass_part;
                    else
                        if (block_index_i == 0)
                            cell_matrix(i,j) += -current_kappa_im*mass_part;
                        else if (block_index_i == 1)
                            cell_matrix(i,j) += current_kappa_im*mass_part;
                    

                }
                if (block_index_i == 0)
                    cell_rhs(i) += dotprod(value_i,rhs_values_re[q_point])*fe_values.JxW(q_point);
                else if (block_index_i == 1)
                    cell_rhs(i) += dotprod(value_i,rhs_values_im[q_point])*fe_values.JxW(q_point);
                
                
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
    Vector<double> diff_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, exact_solution_re,
                                      diff_per_cell, QGauss<dim>(quad_order), VectorTools::L2_norm);
    const double L2_error = diff_per_cell.l2_norm();
    
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("E_re L2 Error", L2_error);
//     convergence_table.add_value("E_im L2 Error", L2_error);
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
    for (unsigned int cycle=0; cycle<3; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            /* Cube mesh */
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (1);
            

            
            /* Read in mesh:
            GridIn<dim> gridin;
            gridin.attach_triangulation(triangulation);
            std::ifstream mesh_file("../meshes/cubeincube_nobc.ucd");
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
            //triangulation.refine_global(2); */
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
