/*-------------ConjGrad.cpp---------------------------------------------------//
*
*              Conjugate Gradient Method
*
* Purpose: This file intends to solve a simple linear algebra equation:
*              Ax = b
*              Where A is a square Matrix of dimension N and x, b are vectors
*              of length N. 
*
*   Notes: This will use the eigen LA library for solving things.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <Eigen/Core>
#include <memory>
#include <cassert>
#include <cmath>

using namespace Eigen;

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const Matrix2d &A, const Vector2d &b, Vector2d &x_0, 
              double thresh);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    double thresh = 0.001;

    Matrix2d a(2,2);
    Vector2d b, x;

    a(0,0) = 4; a(1,0) = 1; a(0,1) = 1; a(1,1) = 3;
    x(0) = 2; x(1) = 1;
    b(0) = 1; b(1) = 2;

    conjgrad(a, b, x, thresh);

    std::cout << "Conjugate Gradient output with threshold " << thresh << '\n';

    std::cout << x(0) << '\t' << x(1) <<'\n';

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function for Conj Gradient -- All the big stuff happens here
void conjgrad(const Matrix2d &A, const Vector2d &b, Vector2d &x_0, 
              double thresh){

    Vector2d r, p;
    double alpha, diff, beta, temp1, temp2;

    // definingin first r
    Vector2d r_0 = b - (A * x_0);

    // setting x arbitrarily high to start
    Vector2d x = x_0 * 4;

    // *grumble... gustorn... grumble*
    Vector2d p_0 = r_0;

    diff = (x - x_0).norm();
    while (diff > thresh){

        // Note, matmul will output a Matrix2d with one element.
        temp1 = r_0.transpose() * r_0;
        temp2 = p_0.transpose() * A * p_0;
        alpha = temp1 / temp2;

        x = x_0 + (p_0 * alpha);
        r = r_0 - (A * alpha * p_0);

        temp1 = r.transpose() * r;
        temp2 = r_0.transpose() * r_0;

        beta = temp1 / temp2;
        
        p = r + (p_0 * beta); 

        diff = (x - x_0).norm();;

        // set all the values to naughts
        x_0 = x;
        r_0 = r;
        p_0 = p;
    }
}

