/*-------------matrix.cpp-----------------------------------------------------//
*
*              matrix.cpp
*
* Purpose: To perform simple Lin. Alg. calculations, such as trace, determinant,
*          and determining the eigenvalues and vectors
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <array>
#include <cmath>

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

const int n = 2;

// finds trace
double trace(std::array<std::array<double, n>, n> matrix);

// finds determinant
double det(std::array<std::array<double, n>, n> matrix); 

// find eigenvalues
std::array<double, n> eigen(std::array<std::array<double, n>, n> matrix); 

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    std::array<std::array<double, 2>, 2> matrix;
    matrix[0][0] = 1.0;
    matrix[0][1] = 3.0;
    matrix[1][0] = 4.0;
    matrix[1][1] = 1.0;

    std::array<double, 2> value = eigen(matrix);
    std::cout << value[0] << '\t' << value[1] << '\n';

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// finds trace
double trace(std::array<std::array<double, n>, n> matrix){
    double trace = 0;

    for (size_t i = 0; i < n; i++){
        trace += matrix[i][i];
    }

    return trace;
}

// finds determinant
double det(std::array<std::array<double, n>, n> matrix){
    double determinant;
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    return determinant;
}

// find eigenvalues
std::array<double, n> eigen(std::array<std::array<double, n>, n> matrix){

    double root, tr;
    std::array<double, n> value;
    tr = trace(matrix);
    root = sqrt(tr * tr - 4 * det(matrix));
    
    value[0] = 0.5 * (-trace(matrix) + root);
    value[1] = 0.5 * (-trace(matrix) - root);

    return value;
}

