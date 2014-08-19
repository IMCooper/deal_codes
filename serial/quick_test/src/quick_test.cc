#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>

#include <iostream>


using namespace dealii;

// MAIN MAXWELL CLASS
/*
template <int dim>
class MaxwellProblem
{
public:
    MaxwellProblem (const unsigned int order);
//    ~MaxwellProblem ();
    void run ();
private:
    FE_Nedelec<dim>          fe;
    unsigned int p_order;
};

template <int dim>
MaxwellProblem<dim>::MaxwellProblem (const unsigned int order)
:
fe (order)
{
    p_order = order;
}

template <int dim>
void MaxwellProblem<dim>::run ()
{
    std::cout << "Order is " << p_order << std::endl;
    std::cout << "Nedelec components: " << fe.n_components() << std::endl;
    std::cout << "DoFs per cell: " << fe.n_dofs_per_cell() << std::endl;
    ~fe();
}
*/
// END MAXWELL CLASS
int main ()
{
    const unsigned int dim=3;
    Timer timer;
    for (unsigned int k=0;k<3;k++)
    {
        timer.start ();
        FE_Nedelec<dim>          fe(k);
        timer.stop ();
        std::cout << "Order " << k << std::endl;
        std::cout << "DoFs per cell: " << fe.n_dofs_per_cell() << std::endl;
        std::cout << "Time: " << timer() << "s" << std::endl;
        timer.reset();
    }
/*
    timer.start ();
    MaxwellProblem<dim> maxwell0(0);
    maxwell0.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
    
    timer.start ();
    MaxwellProblem<dim> maxwell1(1);
    maxwell1.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
    
    timer.start ();
    MaxwellProblem<dim> maxwell2(2);
    maxwell2.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
    
    timer.start ();
    MaxwellProblem<dim> maxwell3(3);
    maxwell3.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
    
    timer.start ();
    MaxwellProblem<dim> maxwell4(4);
    maxwell4.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
    timer.start ();
    MaxwellProblem<dim> maxwell5(5);
    maxwell5.run ();
    timer.stop ();
    std::cout << "Time: " << timer() << "s" << std::endl;
    timer.reset();
 */
    return 0;
}
