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
*          matmag is not for matrices yet. Fix that.
*          ConjGrad function not working or tested yet.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>

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

// Function for addition
array2D matadd(const array2D &A, const array2D &B);
array2D matdiff(const array2D &A, const array2D &B);

// Function for scaling
array2D matscale(const array2D &A, double scale);

// Function for finding the magnitude of a vector... Note not array yet.
double matmag(const array2D &A);

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const array2D &A, const array2D &b, array2D &x_0, double thresh);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    array2D a(2,2), b(2,1);

    a(0,0) = 4; a(1,0) = 1; a(0,1) = 1; a(1,1) = 3;
    b(0,0) = 2; b(0,1) = 1;

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

void conjgrad(const array2D &A, const array2D &b, array2D &x_0, double thresh){

    // Not working. Messily break if someone tries to use it.
    std::cout << "Not working" << '\n';
    assert(thresh < 0);

    array2D r(x_0.rows, x_0.columns), p(x_0.rows,x_0.columns);
    double alpha, diff;

    // definingin first r
    array2D r_0 = matdiff(b, matmul(A,x_0));

    // setting x arbitrarily high to start
    array2D x = matscale(x_0, 4);

    // *grumble... gustorn... grumble*
    array2D p_0 = matscale(r_0,1);
/*
    while (diff > thresh){
    }
*/
}

// Function for scaling
array2D matadd(const array2D &A, const array2D &B){

    if (A.rows != B.rows || A.columns != B.columns){
        std::cout << "incorrect dimensions with addition" << '\n';
        assert(A.rows == B.rows || A.columns == B.columns);
    }

    array2D C(A.rows, B.columns);

    for (size_t i = 0; i < C.rows; i++){
        for (size_t j = 0; j < C.columns; j++){
            C(i, j) = A(i, j) + B(i, j);
        }
    }

    return C;
}

// Function for scaling
array2D matdiff(const array2D &A, const array2D &B){

    if (A.rows != B.rows || A.columns != B.columns){
        std::cout << "incorrect dimensions with addition" << '\n';
        assert(A.rows == B.rows || A.columns == B.columns);
    }

    array2D C(A.rows, B.columns);

    for (size_t i = 0; i < C.rows; i++){
        for (size_t j = 0; j < C.columns; j++){
            C(i, j) = A(i, j) - B(i, j);
        }
    }

    return C;
}

// Function for scaling
array2D matscale(const array2D &A, double scale){

    array2D C(A.rows, A.columns);

    for (size_t i = 0; i < C.rows; i++){
        for (size_t j = 0; j < C.columns; j++){
            C(i, j) = A(i, j) * scale;
        }
    }

    return C;
}

// Function for finding magnitude for a single columned array
// Note: We should probably set this function to work for all types of matrices
double matmag(const array2D &A){
    double mag = 0;

    if (A.columns > 1){
        std::cout << "Not implemented yet for larger than 1 column." << '\n';
        assert(A.columns < 1);
    }

    for (size_t i = 0; i < A.rows; i++){
        mag += A(i,0)*A(i,0);
    }

    mag = sqrt(mag);
    return mag;
}
