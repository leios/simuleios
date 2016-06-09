/*-------------Lanczos.cpp----------------------------------------------------//
*
* Purpose: To diagonalize a random matrix using the Lanczos algorithm
*
*   Notes: Compile with (for Arch systems):
*              g++ -I /usr/include/eigen3/ Lanczos.cpp
*
*-----------------------------------------------------------------------------*/
 
#include <iostream>
#include <Eigen/Core>
#include <random>
#include <vector>

using namespace Eigen;

// Function for the lanczos algorithm, returns Tri-diagonal matrix
MatrixXd lanczos(MatrixXd d_matrix);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    int size = 200;
    MatrixXd d_matrix(size,size);

    // set up random device
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0,1);

    for (size_t i = 0; i < d_matrix.rows(); ++i){
        for (size_t j = 0; j < i; ++j){
            d_matrix(i,j) = dist(gen);
            d_matrix(j,i) = d_matrix(i,j);
        }
    }

    MatrixXd Q = lanczos(d_matrix);

    for (size_t i = 0; i < Q.rows(); ++i){
        for (size_t j = 0; j < Q.cols(); ++j){
            std::cout << Q(i, j) << '\t';
        }
        std::cout << '\n';
    }

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function for the lanczos algorithm, returns Tri-diagonal matrix
MatrixXd lanczos(MatrixXd d_matrix){

    // Creating random device
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0,1); 

    // Defining values
    double threshold = 0.01;
    int j = 0, j_tot = 5;
    int size = d_matrix.rows();

    // Setting beta arbitrarily large for now 
    double beta = 10;

    // generating the first rayleigh vector
    // alpha is actually just a double... sorry about that.
    MatrixXd rayleigh(d_matrix.rows(),1), q(d_matrix.rows(),1),
             alpha(1, 1);
    MatrixXd identity = MatrixXd::Identity(d_matrix.rows(), d_matrix.cols());

    // krylov is the krylovian subspace... Note, there might be a dynamic way to
    // do this. Something like:
    //std::vector <MatrixXd> krylov;
    MatrixXd krylov(d_matrix.rows(), j_tot);

    for (size_t i = 0; i < size; ++i){
        rayleigh(i) = dist(gen);
    }

    //std::cout << rayleigh << '\n';

    //while (beta > threshold){
    for (size_t i = 0; i < j_tot; ++i){
        beta = rayleigh.norm();
        //std::cout << "beta is: \n" << beta << '\n';

        q = rayleigh / beta;
        //std::cout << "q is: \n" << q << '\n';

        alpha = q.transpose() * d_matrix * q;
        //std::cout << "alpha is \n" << alpha << '\n';

        if (j == 0){
            rayleigh = (d_matrix - alpha(0,0) * identity) * q;
        }
        else{
            rayleigh = (d_matrix - alpha(0,0) * identity) * q 
                       - beta * krylov.col(j - 1);

        }
        //std::cout << "rayleigh is: \n" << rayleigh <<'\n';
        //std::cout << "i is: " << i << '\n';

        //krylov.push_back(q);
        krylov.col(j) = q;
        j = j+1;
        std::cout << j << '\n';
    }

    MatrixXd krylov_id = krylov.transpose() * krylov;
    std::cout << "The identity matrix from the krylov subspace is: \n" 
              << krylov_id << '\n';

    MatrixXd T(j_tot,j_tot);
    T = krylov.transpose() * d_matrix * krylov;

    // Setting values to 0 if they are close...
    for (size_t i = 0; i < T.rows(); ++i){
        for (size_t j = 0; j < T.cols(); ++j){
            if (T(i,j) < 0.00001){
                T(i,j) = 0;
            }
        }
    }

    return T;
}
