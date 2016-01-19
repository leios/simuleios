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
#define DIMS (DOF + 2)

// Note that the array has two additional column elements:
//     1. particle potential
//     2. ID

typedef Eigen::Matrix<double, // typename Scalar
   SIZE, // int RowsAtCompileTime,
   DIMS, // int ColsAtCompileTime,
   0> // int Options = 0,
   MatrixPSIP;

struct H3plus { 
    MatrixPSIP pos;
    double Vref, dt, Energy;
    int psipnum, id;
};

// Populate a distribution of particles for QMC
void populate(H3plus& state);

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
void find_weights(H3plus& state);

// Branching scheme
void branch(H3plus& state);

// Random walking of matrix of position created in populate
void diffuse(H3plus& state, std::ostream& output);

// output certain elements in array
void arrayout(H3plus& state, int length);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    std::ofstream output("out.dat", std::ostream::out);
    H3plus state;

    state.Vref = 0;
    state.dt = 0.001;
    state.psipnum = 1000;
    state.Energy = 0;
    state.id = 0;

    populate(state);

    std::cout << state.Vref << '\t' << state.psipnum << '\n';

    /*
    for (size_t i = 0; i < state.psipnum; i++){
        std::cout << " final element is: " << state.pos(i,DIMS-2) << '\n';
    }
    */

    diffuse(state, output);

    //output << state.pos << '\n';

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Populate a distribution of particles for QMC
// Unlike Anderson, we are initilizing each psip randomly from a distribution.
// This might scre things up, because
void populate(H3plus& state){

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0,1.0);

    double radius = 1.8;

    /*
    for (size_t i = 0; i < state.psipnum; i++){
        for (size_t j = 0; j < state.pos.cols()-1; j++){
            state.pos(i,j) = distribution(generator);
        }
    }
    */

    // Initializing position and ID, potential in find_weights beneath
    for (size_t i = 0; i < state.psipnum; i++){
        state.pos(i,0) = radius;
        state.pos(i,1) = radius;
        state.pos(i,2) = radius;
        state.pos(i,3) = -radius;
        state.pos(i,4) = -radius;
        state.pos(i,5) = -radius;
        state.pos(i,DIMS-1) = state.id;
        state.id++;
    }

    find_weights(state);

    /*
    for (size_t i = 0; i < state.psipnum; i++){
        std::cout << state.pos(i,DIMS-2) << '\n';
    }
    */

}

// Calculates energy of a configuration and stores it into the final element of 
// the MatrixPSIP
// Note: When calculating the potential, I am not sure whether we need to use
//       absolute value of distance or the distance, itself.
// Note: Inefficient. Can calculate energy on the fly with every generation of 
//       psip. Think about it.
// Note: Vref will be a dummy variable for now.
void find_weights(H3plus& state){

    double dist, pot, pot_tot = 0;

    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(0,1);


    // Note that this is specific to the Anderson paper
    // Finding the distance between electrons, then adding the distances
    // from the protons to the electrons.
    for (size_t i = 0; i < state.psipnum; i++){
        pot = 0;
        dist = sqrt(((state.pos(i,0) - state.pos(i,3)) * 
                    (state.pos(i,0) - state.pos(i,3))) +
                    ((state.pos(i,1) - state.pos(i,4)) * 
                    (state.pos(i,1) - state.pos(i,4))) +
                    ((state.pos(i,2) - state.pos(i,5)) * 
                    (state.pos(i,2) - state.pos(i,5))));
        for (size_t j = 0; j < state.pos.cols() - 2; j++){
            pot -= 1.0 / std::abs(state.pos(i,j));
        }
        pot += 1.0 / dist;
        state.pos(i,DIMS-2) = pot;

        pot_tot += pot;

        // Comment these in for debugging the branching step
        /*
        if (i == 0){state.pos(i,DIMS-2) = 0;}
        if (i == 1){state.pos(i,DIMS-2) = 3;}
        if (i == 2){state.pos(i,DIMS-2) = 3;}
        if (i == 3){state.pos(i,DIMS-2) = 2;}
        if (i == 4){state.pos(i,DIMS-2) = 0;}
        if (i == 5){state.pos(i,DIMS-2) = 0;}
        if (i == 6){state.pos(i,DIMS-2) = 0;}
        */
        //std::cout << state.pos(i,DIMS-2) << '\n';
    }

    // defining the new reference potential to psipnum down.
    state.Energy = pot_tot / state.psipnum;
    state.Vref = state.Energy
                 - ((state.psipnum - 1000) / (1000 * state.dt));

    for (size_t i = 0; i < state.psipnum; i++){
        state.pos(i,DIMS-2) = (int)(1-(state.pos(i,DIMS-2) - state.Vref) 
                                    * state.dt
                              + distribution(gen));
        if (state.pos(i,DIMS-2) > 3){
            state.pos(i,DIMS-2) = 3;
        }

    }
}

// Branching scheme
void branch(H3plus& state){

    find_weights(state);

    /*
    for (size_t i = 0; i < state.psipnum; i++){
        std::cout << state.pos(i,DIMS-2) << '\n';
    }
    */

    int variable, offset = 0, psip_old = state.psipnum, births = 0, tmpi = 0;

    for (size_t i = 0; i < psip_old + births; i++){

        //std::cout << i << '\n';

        variable = state.pos(i, DIMS-2);

        switch (variable){
            // Destruction
            case 0: state.psipnum--;
                    break;

            // Creation of 1
            case 2: state.psipnum++;
                    births++;
                    for (size_t j = 0; j < state.pos.cols() - 2; j++){
                        state.pos(psip_old+births-1,j) = state.pos(i,j);
                    }
                    //std::cout << "writing: " << i << " to " 
                    //          << psip_old+births-1 << '\n';
                    state.pos(psip_old+births-1,DIMS-2) = 1;
                    state.pos(psip_old+births-1,DIMS-1) = state.id;
                    state.id++;
                    break;

            // Creation of 2
            case 3: state.psipnum += 2;
                    births += 2;
                    for (size_t k = 0; k < 2; k++){
                        for (size_t j = 0; j < state.pos.cols() - 2; j++){
                            state.pos(psip_old-k+births-1, j) = state.pos(i, j);
                            //std::cout << state.pos(psip_old-k+births-1, j) 
                            //          << '\n';
                        }
                        state.pos(psip_old-k+births-1,DIMS-2) = 1;
                        state.pos(psip_old-k+births-1,DIMS-1) = state.id;
                        state.id++;
                        //std::cout << "writing: " << i << " to " 
                        //          << psip_old-k+births-1 << '\n';
                    }
                    break;

        }

    }

    //arrayout(state, state.psipnum);

    // Adjustment for offset
    // Note: Account for the situation where offset is greater than arraysize
    for (size_t i = 0; i < SIZE; i++){
        if (state.pos(i,DIMS-2) != 0){
            for (size_t j = 0; j < state.pos.cols(); j++){
                state.pos(tmpi,j) = state.pos(i,j);
            }
            tmpi++;
        }
        if (i > state.psipnum){
            for (size_t j = 0; j < state.pos.cols(); j++){
                state.pos(i,j) = 0;
            }

        }
    }

}

// Random walking of matrix of position created in populate
// Step 1: Move particles via 6D random walk
// Step 2: Destroy and create particles as need based on Anderson
// Step 3: check energy, end if needed.
void diffuse(H3plus& state, std::ostream& output){

    // Let's initialize the randomness
    std::default_random_engine gen;
    std::normal_distribution<double> gaussian(0,1);

    double diff = 1, Vsave = 0;

    // For now, I am going to set a definite number of timesteps
    // This will be replaced by a while loop in the future.
    for (size_t t = 0; t < 10000; t++){
    //while (diff > 0.01){
        Vsave = state.Vref;
        for (size_t i = 0; i < state.psipnum; i++){
            for (size_t j = 0; j < state.pos.cols() - 1; j++){
                state.pos(i, j) += sqrt(state.dt) * gaussian(gen);
            }
        }
        branch(state);
        diff = sqrt((Vsave - state.Vref)*(Vsave - state.Vref));
        std::cout << state.Vref << '\t' << state.psipnum << '\n';
        if (t % 1000 == 0){
            output << state.pos << '\n' << '\n' << '\n';
        }
    }

}

// output certain elements in array
void arrayout(H3plus& state, int length){

    for (size_t i = 0; i < length; i++){
        for (size_t j  = 0; j < 2; j++){
            std::cout << state.pos(i,DIMS - 1 - j) << '\t';
        }
        std::cout << '\n';
    } 
}
