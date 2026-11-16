// Practical 2c: OpenMP Matrix Multiplication  (C = A x B)
// Compile: g++ -fopenmp 2c_matrix_multiplication.cpp -o 2c
#include <iostream>
#include <vector>
#include <omp.h>

using Matrix = std::vector<std::vector<int>>;

int main() {
    int m, n, p;
    std::cout << "Enter the number of rows in Matrix 1: ";
    std::cin >> m;
    std::cout << "Enter the number of columns in Matrix 1 (= rows in Matrix 2): ";
    std::cin >> n;
    std::cout << "Enter the number of columns in Matrix 2: ";
    std::cin >> p;

    Matrix A(m, std::vector<int>(n));
    Matrix B(n, std::vector<int>(p));
    Matrix C(m, std::vector<int>(p, 0));

    std::cout << "<-- Input matrix 1 row-wise -->\n";
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            std::cin >> A[i][j];

    std::cout << "<-- Input matrix 2 row-wise -->\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            std::cin >> B[i][j];

    // Rows of C are divided among the threads. Loop variables declared inside
    // the loop are automatically private to each thread.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }

    std::cout << "The output after Matrix Multiplication is:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++)
            std::cout << C[i][j] << "\t";
        std::cout << "\n";
    }
    return 0;
}
