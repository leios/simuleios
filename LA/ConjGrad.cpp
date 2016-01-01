/*-------------ConjGrad.cpp---------------------------------------------------//
*
*              Conjugate Gradient Method
*
* Purpose: This file intends to solve a simple linear algebra equation:
*              Ax = b
*              Where A is a square matrix of dimension N and x, b are vectors
*              of length N. 
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <memory>

// Struct to hold 2d array
struct array2D{
    // initial pointers to array
    std::unique_ptr<double[]> data;
    size_t n;

    // this defines our interface with array creation
    array2D(size_t n) : data(new double[n*n]), n(n) {
        for(size_t i = 0; i < n*n; i++){ data[i] = 0; }
    }

    // These are interfaces for array calls. Basically setting the row length.
    double operator() (size_t row, size_t col) const {
        return data[row * n + col];
    }

    double& operator() (size_t row, size_t col) {
        return data[row * n + col];
    }
};

// Smaller function for Matrix multiply
array2D matmul(const array2D &A, const array2D &B);

// Function for Conj Gradient -- All the big stuff happens here
std::vector<double>  conjgrad(const array2D &A, const std::vector<double> &b, 
                               std::vector<double> &x);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    array2D test(10);
    std::cout << test(0,10) << '\n';
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Smaller function for Matrix multiply
array2D matmul(const array2D &A, const array2D &B){
    array2D C(A.n);

    return C;
}

// Function for Conj Gradient -- All the big stuff happens here
std::vector<double>  conjgrad(const array2D &A, const std::vector<double> &b, 
                              std::vector<double> &x){
    return x;
}

