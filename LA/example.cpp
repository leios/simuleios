#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

#include "matrix.h"

// A more robust version of solving quadtratic equations using floating
// point numbers. Assumes that there's a solution
static std::pair<double, double> solve_quadtratic(double a, double b, double c) {
    auto d = b * b - 4.0 * a * c;
    assert(d >= 0.0);
    double t = -0.5 * (b + std::copysign(1.0, b) * std::sqrt(d));
    return {t / a, c / t};
}

static std::pair<double, double> eigen_values(const matrix<2>& matrix) {
    return solve_quadtratic(1.0, trace(matrix), det(matrix));
}

// Eigen values are the main example, this is here just for fun.
static void random_showcase() {
    matrix<2> m = { 4,  3,
                   -2, -3};

    // It's very natural to chain matrix operations: you do them as if they
    // were built-in types
    matrix<2> a, b, c, d, e, f; // Just so the examples actually look sensible
    a = b = c = d = e = f = m;

    // Calculate the result of an arbitrary expression. The only operator not
    // present is divison.
    auto result = 0.75 * a + b * (c - d * 1.3) + identity<2>();
    std::cout << result << "Determinant: " << det(result) << "\n\n";

    // You can do arithmetic assigments as you always would:
    a += b;

    // Indexing works using operator()
    std::cout << a
              << "The value at row 1, column 0: " << a(1, 0) << '\n';
}

int main() {
    matrix<2> m = { 4,  3,
                   -2, -3};

    std::cout << "Eigen Values:\n=============\n";
    auto ev = eigen_values(m);
    std::cout << m
              << "Eigen values: " << ev.first << ", " << ev.second << "\n";

    std::cout << "\nRandom Showcase:\n================\n";
    random_showcase();
    return 0;
}
