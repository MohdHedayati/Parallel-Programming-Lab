// Practical 2f: OpenMP - calculate Pi
// Pi = integral from 0 to 1 of 4/(1+x^2) dx, approximated with the midpoint rule.
// Compile: g++ -fopenmp 2f_pi_calculation.cpp -o 2f
#include <iostream>
#include <iomanip>
#include <omp.h>

int main() {
    const int numSteps = 10000;
    const double step = 1.0 / numSteps;
    double sum = 0.0;

    // reduction(+:sum): each thread keeps a private partial sum; they are added
    // together when the loop ends. Faster than a critical section on every step.
    #pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < numSteps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "The Value of PI is " << pi << "\n";
    return 0;
}
