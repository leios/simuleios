/*-------------Anderson.cpp---------------------------------------------------//
*
*              Anderson.cpp -- Diffusion Monte Carlo for Schroedy
*
* Purpose: Implement the quantum Monte Carlo (diffusion) by Anderson:
*             http://www.huy-nguyen.com/wp-content/uploads/QMC-papers/Anderson-JChemPhys-1975.pdf
*          For H3, with 3 protons and 2 electrons
*
*    Note: This algorithm may be improved by later work
*          A "psip" is an imaginary configuration of electrons in space.
*          Requires c++11 for random and Eigen for matrix
*          Protons assume to be at origin
*          I might have been a little overzealous about Eigen usage...
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <Eigen/Core>

#define PSIPNUM 1000
#define DIMS 6

typedef Eigen::Matrix<double, // typename Scalar
   PSIPNUM, // int RowsAtCompileTime,
   DIMS, // int ColsAtCompileTime,
   0> // int Options = 0,
   MatrixPSIP;

// Populate a distribution of particles for QMC
void populate(MatrixPSIP& pos);

// Random walking of matrix of position created in populate
//void diffuse(MatrixPSIP &pos, double Vref);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    MatrixPSIP pos(PSIPNUM, DIMS);

    populate(pos);

    std::cout << pos << '\n';
    
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Populate a distribution of particles for QMC
// Unlike Anderson, we are initilizing each psip randomly from a distribution.
// This might screw things up, because
void populate(MatrixPSIP &pos){

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    for (size_t i = 0; i < pos.rows(); i++){
        for (size_t j = 0; j < pos.cols(); j++){
           pos(i,j) = distribution(generator);
        }
    }

}

// Random walking of matrix of position created in populate
//void diffuse(Matrix2d &pos, double Vref){

    
//}
