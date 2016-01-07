/*-------------Anderson.cpp---------------------------------------------------//
*
*              Anderson.cpp -- Diffusion Monte Carlo for Schroedy
*
* Purpose: Implement the quantum Monte Carlo (diffusion) by Anderson:
*             http://www.huy-nguyen.com/wp-content/uploads/QMC-papers/Anderson-JChemPhys-1975.pdf
*          For H3+, with 3 protons and 2 electrons
*
*    Note: This algorithm may be improved by later work
*          A "psip" is an imaginary configuration of electrons in space.
*          Requires c++11 for random and Eigen for matrix
*          Protons assume to be at origin
*          A lot of this algorithm is more easily understood here:
*              http://www.thphys.uni-heidelberg.de/~wetzel/qmc2006/KOSZ96.pdf
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <Eigen/Core>

#define DOF 6
#define SIZE 2000
#define DIMS (DOF + 1)

typedef Eigen::Matrix<double, // typename Scalar
   SIZE, // int RowsAtCompileTime,
   DIMS, // int ColsAtCompileTime,
   0> // int Options = 0,
   MatrixPSIP;

struct H3plus { 
    MatrixPSIP pos;
    double Vref;
    int psipnum;
};

// Populate a distribution of particles for QMC
void populate(H3plus& state);

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
void find_weights(H3plus& state);

// Branching scheme
void branch(H3plus& state);

// Random walking of matrix of position created in populate
void diffuse(H3plus& state);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    H3plus state;

    state.Vref = 1;

    populate(state);

    std::cout << state.pos << '\n';
    
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Populate a distribution of particles for QMC
// Unlike Anderson, we are initilizing each psip randomly from a distribution.
// This might screw things up, because
void populate(H3plus& state){

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    for (size_t i = 0; i < state.psipnum; i++){
        for (size_t j = 0; j < state.pos.cols()-1; j++){
            state.pos(i,j) = distribution(generator);
        }
    }

    find_weights(state);

}

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
// Note: When calculating the potential, I am not sure whether we need to use
//       absolute value of distance or the distance, itself.
// Note: Inefficient. Can calculate energy on the fly with every generation of 
//       psip. Think about it.
// Note: Vref will be a dummy variable for now.
void find_weights(H3plus& state){

    double dist;

    for (size_t i = 0; i < state.psipnum; i++){
        state.pos(i,DIMS-1) = 0;
        dist = sqrt((state.pos(i,0) - state.pos(i,3)) * 
                    (state.pos(i,0) - state.pos(i,3)) +
                    (state.pos(i,1) - state.pos(i,4)) * 
                    (state.pos(i,1) - state.pos(i,4)) +
                    (state.pos(i,2) - state.pos(i,5)) * 
                    (state.pos(i,2) - state.pos(i,5)));
        for (size_t j = 0; j < state.pos.cols() - 1; j++){
            state.pos(i, DIMS-1) = exp(-(state.Vref - (1.0 / state.pos(i,j))));
        }
        state.pos(i,DIMS-1) += 1.0 / dist;
    }
}

// Branching scheme
void branch(H3plus& state){

    int variable, offset = 0, psip_old = state.psipnum;

    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(0,1);

    for (size_t i = 0; i < state.pos.rows(); i++){
        variable = (unsigned int)(state.pos(i,DIMS) * distribution(gen));

        if (variable > 3){
            variable = 3;
        }

        switch (variable){
            // Destruction
            case 0: state.psipnum--;
                    offset++;
                    break;

            // Nothing
            // case 1: 

            // Creation of 1
            case 2: state.psipnum++;
                    for (size_t j = 0; j < state.pos.cols(); j++){
                        state.pos(psip_old, j) = state.pos(i+offset, j);
                    }
                    break;

            // Creation of 2
            case 3: state.psipnum += 2;
                    for (size_t k = 0; k < 2; k++){
                        for (size_t j = 0; j < state.pos.cols(); j++){
                            state.pos(psip_old - k, j) = state.pos(i, j);
                        }
                    }
                    break;

        }
    }

}

// Random walking of matrix of position created in populate
// Step 1: Move particles via 6D random walk
// Step 2: Destroy and create particles as need based on Anderson
// Step 3: check energy, end if needed.
void diffuse(H3plus& state){

    double dt = 0.1;

    populate(state);

    // Let's initialize the randomness
    std::default_random_engine gen;
    std::normal_distribution<double> gaussian(0,1);

    // For now, I am going to set a definite number of timesteps
    // This will be replaced by a while loop in the future.
    for (size_t t = 0; t < 100; t++){
        for (size_t i = 0; i < state.psipnum; i++){
            for (size_t j = 0; j < state.pos.cols(); j++){
                state.pos(i, j) += sqrt(dt) * gaussian(gen);
            }
        }
    }

    branch(state);
}
