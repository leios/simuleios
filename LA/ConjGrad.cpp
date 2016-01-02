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
#include <cassert>

// Struct to hold 2d array
struct array2D{
    // initial pointers to array
    std::unique_ptr<double[]> data;
    size_t rows, columns;

    // defines our interface with array creation and creates struct memebers
    array2D(size_t n, size_t m) : data(new double[(n)*(m)]), 
                                  rows(n), 
                                  columns(m) {
        for(size_t i = 0; i < n*m; i++){ data[i] = 0; }
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
void conjgrad(const array2D &A, const array2D &b, array2D &x);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    array2D a(2,2), b(2,1);

    array2D c = matmul(a, b);

    for (size_t i = 0; i < c.rows; i++){
        for (size_t j = 0; j < c.columns; j++){
            std::cout << c(i,j) << '\t';
        }
        std::cout << '\n';
    }
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Smaller function for Matrix multiply
array2D matmul(const array2D &A, const array2D &B){

    if (A.columns != B.rows){
        std::cout << "Incorrect inner dimensions for multiplication." << '\n';
        assert(A.columns == B.rows);
    }

    // Check to make sure we are creating C with an appropriate size
    array2D C(A.rows, B.columns);

    for (size_t i = 0; i < C.rows; i++){
        for (size_t j = 0; j < C.columns; j++){
            for (size_t k = 0; k < A.columns; k++){
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return C;
}

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const array2D &A, const array2D &b, array2D &x){
}

