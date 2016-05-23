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
double magnitude(const array2D &A);

// Function for finding the transpose
array2D transpose(const array2D &A);

// For bpaf because he said I was lazy
void matprint(const array2D &A);

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const array2D &A, const array2D &b, array2D &x_0, double thresh);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    double thresh = 0.001;

    array2D a(2,2), b(2,1), x(2,1);

    a(0,0) = 4; a(1,0) = 1; a(0,1) = 1; a(1,1) = 3;
    x(0,0) = 2; x(0,1) = 1;
    b(0,0) = 1; b(0,1) = 2;

    array2D c = matmul(a, x);

    std::cout << "Simple matrix multiplication check:" << '\n';
    matprint(c);

    std::cout << '\n';

    conjgrad(a, b, x, thresh);

    std::cout << "Conjugate Gradient output with threshold " << thresh << '\n';
    matprint(x);

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
        std::cout << "incorrect dimensions with difference" << '\n';
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
double magnitude(const array2D &A){
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

// Function for finding the transpose
array2D transpose(const array2D &A){
    array2D C(A.columns, A.rows);

    for (size_t i = 0; i < A.rows; i++){
        for (size_t j = 0; j < A.columns; j++){
            C(j, i) = A(i, j);
        }
    }
    return C;
}

void matprint(const array2D &A){
    for (size_t i = 0; i < A.rows; i++){
        for (size_t j = 0; j < A.columns; j++){
            std::cout << A(i,j) << '\t';
        }
        std::cout << '\n';
    }

}

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const array2D &A, const array2D &b, array2D &x_0, double thresh){

    array2D r(x_0.rows, x_0.columns), p(x_0.rows,x_0.columns), 
            temp1(1,1), temp2(1,1);
    double alpha, diff, beta;

    // definingin first r
    array2D r_0 = matdiff(b, matmul(A,x_0));

    // setting x arbitrarily high to start
    array2D x = matscale(x_0, 4);

    // *grumble... gustorn... grumble*
    array2D p_0 = matscale(r_0,1);

    diff = magnitude(matdiff(x, x_0));
    while (diff > thresh){

        // Note, matmul will output a array2D with one element.
        temp1 = matmul(transpose(r_0), r_0);
        temp2 = matmul(transpose(p_0), matmul(A, p_0));
        alpha = temp1(0,0) / temp2(0,0);

        x = matadd(x_0, matscale(p_0,alpha));
        r = matadd(r_0, matmul(matscale(A,-alpha),p_0));

        temp1 = matmul(transpose(r),r);
        temp2 = matmul(transpose(r_0),r_0);

        beta = temp1(0,0) / temp2(0,0);
        
        p = matadd(r, matscale(p_0,beta)); 

        diff = magnitude(matdiff(x, x_0));

        // set all the values to naughts
        x_0 = matscale(x,1);
        r_0 = matscale(r,1);
        p_0 = matscale(p,1);
    }
}

