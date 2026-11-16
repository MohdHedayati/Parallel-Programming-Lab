// Practical 2e: OpenMP - largest element in an array
// Compile: g++ -fopenmp 2e_largest_element.cpp -o 2e
#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>

int main() {
    int n;
    std::cout << "Enter the number of elements: ";
    std::cin >> n;

    std::vector<int> arr(n);
    std::cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++)
        std::cin >> arr[i];

    int currentMax = INT_MIN;

    // reduction(max:...) gives each thread its own private max, then combines
    // them at the end. No critical section is needed, so it runs truly in parallel.
    #pragma omp parallel for reduction(max : currentMax)
    for (int i = 0; i < n; i++) {
        if (arr[i] > currentMax)
            currentMax = arr[i];
    }

    std::cout << "The Maximum Element is: " << currentMax << "\n";
    return 0;
}
