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

namespace testy
{
    using namespace dealii;
    const double constant_PI = numbers::PI;
    double var_foo = 0.0;
    
    namespace testy_data
    {
        double var_foo = 1.0;
    }
    class testing
    {
    public:
        void print_out();
    private:
        double copy_to_foo=10.0;
    };
    void testing::print_out()
    {
        std::cout << "Pi is: " << constant_PI << std::endl;
        std::cout << "var_foo was: " << var_foo << std::endl;
        var_foo = copy_to_foo;
        std::cout << "var_foo is now: " << var_foo << std::endl;
    }
}

int main()
{
    using namespace testy;
    testing x;
    x.print_out();
    return 0;
}