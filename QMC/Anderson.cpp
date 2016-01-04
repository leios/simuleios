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
#define DIMS 7

typedef Eigen::Matrix<double, // typename Scalar
   PSIPNUM, // int RowsAtCompileTime,
   DIMS, // int ColsAtCompileTime,
   0> // int Options = 0,
   MatrixPSIP;

// Populate a distribution of particles for QMC
void populate(MatrixPSIP& pos);

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
void find_energy(MatrixPSIP& pos);

// Random walking of matrix of position created in populate -- COMING SOON
// void diffuse(MatrixPSIP& pos, double Vref);

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
        for (size_t j = 0; j < pos.cols()-1; j++){
            pos(i,j) = distribution(generator);
        }
    }

    find_energy(pos);

}

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
// Note: When calculating the potential, I am not sure whether we need to use
//       absolute value of distance or the distance, itself.
// Note: Inefficient. Can calculate energy on the fly with every generation of 
//       psip. Think about it.
void find_energy(MatrixPSIP& pos){

    double dist;

    for (size_t i = 0; i < pos.rows(); i++){
        pos(i,DIMS-1) = 0;
        dist = sqrt((pos(i,0) - pos(i,3)) * (pos(i,0) - pos(i,3)) +
                    (pos(i,1) - pos(i,4)) * (pos(i,1) - pos(i,4)) +
                    (pos(i,2) - pos(i,5)) * (pos(i,2) - pos(i,5)));
        for (size_t j = 0; j < pos.cols() - 1; j++){
            pos(i, DIMS-1) -= (1.0 / pos(i,j));
        }
        pos(i,DIMS-1) += 1.0 / dist;
    }
}


// Random walking of matrix of position created in populate
// Step 1: Move particles via 6D random walk
// Step 2: Destroy and create particles as need based on Anderson
// Step 3: check energy, end if needed.
/*
void diffuse(MatrixPSIP &pos, double Vref){

    // For now, I am going to set a definite number of timesteps
    // This will be replaced by a while loop in the future.
    for (size_t t = 0; t < 100; t++){
        for (size_t i = 0; i < pos.rows(); i++){
            for (size_t j = 0; j < pos.cols(); j++){
            }
        }
    }
}
*/
