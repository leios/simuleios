/*-------------ConjGrad.cpp---------------------------------------------------//
*
*              Conjugate Gradient Method
*
* Purpose: This file intends to solve a simple linear algebra equation:
*              Ax = b
*              Where A is a square matrix of dimension N and x, b are vectors
*              of length N. 
*
*   Notes: We can write this entire code without std:: vector. Think about it
*          Need to make sure we are indexing arrays appropriately.
*          Think about using LAPACK / BLAS
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <memory>

// Struct to hold 2d array
struct array2D{
    // initial pointers to array
    std::unique_ptr<double[]> data;
    size_t rows, columns;

    // defines our interface with array creation and creates struct memebers
    array2D(size_t n, size_t m) : data(new double[n*m]), rows(n), columns(m) {
        for(size_t i = 0; i < n*m; i++){ data[i] = i; }
    }

    // These are interfaces for array calls. Basically setting the row length.
    double operator() (size_t row, size_t col) const {
        return data[row * columns + col];
    }

    double& operator() (size_t row, size_t col) {
        return data[row * columns + col];
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

    array2D test(10, 4);
    std::cout << test(1,4) << '\n';
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Smaller function for Matrix multiply
array2D matmul(const array2D &A, const array2D &B){
    // Check to make sure we are creating C with an appropriate size
    array2D C(A.rows, B.columns);

    return C;
}

// Function for Conj Gradient -- All the big stuff happens here
std::vector<double>  conjgrad(const array2D &A, const std::vector<double> &b, 
                              std::vector<double> &x){
    return x;
}

