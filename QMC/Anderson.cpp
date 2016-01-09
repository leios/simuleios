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
#include <fstream>

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

    std::ofstream output("out.dat", std::ostream::out);
    H3plus state;

    state.Vref = 1;
    state.psipnum = 4;

    populate(state);
    diffuse(state);

    output << state.pos << '\n';

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

    int variable, offset = 0, psip_old = state.psipnum, births = 0;

    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(0,1);

    for (size_t i = 0; i < psip_old + births; i++){
        //variable = (unsigned int)(state.pos(i,DIMS-1) * distribution(gen));
        if (state.pos(i,DIMS-1) == 0){
            variable = 0;
        } 
        else{variable = 1;}

        if (variable > 3){
            variable = 3;
        }

        if (i == 0){variable = 0;}
        if (i == 1){variable = 3;}
        if (i == 2){variable = 2;}
        if (i == 3){variable = 1;}

        switch (variable){
            // Destruction
            case 0: state.psipnum--;
                    offset++;
                    for (size_t j = 0; j < state.pos.cols(); j++){
                        state.pos(i, j) = state.pos(i+offset, j);
                    }
                    std::cout << "deleting particle" << '\t' << i << '\n';
                    break;

            case 1: for (size_t j = 0; j < state.pos.cols(); j++){
                        state.pos(i,j) = state.pos(i+offset,j);
                    }
                    break;

            // Creation of 1
            case 2: state.psipnum++;
                    births++;
                    for (size_t j = 0; j < state.pos.cols() - 1; j++){
                        state.pos(psip_old+births-1,j) = state.pos(i,j);
                        state.pos(i,j) = state.pos(i+offset,j);
                    }
                    state.pos(psip_old+births-1, DIMS-1) = 1;
                    state.pos(i,DIMS-1) = state.pos(i+offset, DIMS-1);
                    break;

            // Creation of 2
            case 3: state.psipnum += 2;
                    births += 2;
                    for (size_t k = 0; k < 2; k++){
                        for (size_t j = 0; j < state.pos.cols() - 1; j++){
                            state.pos(psip_old-k+births-1, j) = state.pos(i, j);
                            std::cout << state.pos(psip_old-k+births-1, j) << '\n';
                            state.pos(i,j) = state.pos(i+offset,j); 
                        }
                        state.pos(psip_old-k+births, DIMS-1) = 1;
                        state.pos(i,DIMS-1) = state.pos(i+offset,DIMS-1);
                        std::cout << "writing: " << psip_old-k+births-1 << '\n';
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
    for (size_t t = 0; t < 1; t++){
        for (size_t i = 0; i < state.psipnum; i++){
            for (size_t j = 0; j < state.pos.cols(); j++){
                state.pos(i, j) += sqrt(dt) * gaussian(gen);
            }
        }
        branch(state);
    }

}
