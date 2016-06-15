/*-------------Lanczos.cpp----------------------------------------------------//
*
* Purpose: To diagonalize a random matrix using the Lanczos algorithm
*
*   Notes: Compile with (for Arch systems):
*              g++ -I /usr/include/eigen3/ Lanczos.cpp
*          0's along the prime diagonal. I don't know what this means.
*
*-----------------------------------------------------------------------------*/
 
#include <iostream>
#include <Eigen/Core>
#include <random>
#include <vector>
#include <math.h>

using namespace Eigen;

// Function for the lanczos algorithm, returns Tri-diagonal matrix
MatrixXd lanczos(MatrixXd d_matrix);

// Function for QR decomposition
MatrixXd qrdecomp(MatrixXd Tridiag);

// Function to return sign of value (signum function)
int sign(double value);

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

    MatrixXd Tridiag = lanczos(d_matrix);

    for (size_t i = 0; i < Tridiag.rows(); ++i){
        for (size_t j = 0; j < Tridiag.cols(); ++j){
            std::cout << Tridiag(i, j) << '\t';
        }
        std::cout << '\n';
    }

    MatrixXd Q = qrdecomp(Tridiag);

    std::cout << Q << '\n';

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

// Function for QR decomposition
// Because we only need Q for the power method, I will retun only Q
MatrixXd qrdecomp(MatrixXd Tridiag){
    // Q is and orthonormal vector => Q'Q = 1
    MatrixXd Q = MatrixXd::Identity(Tridiag.rows(), Tridiag.cols());

    // R is the upper triangular matrix
    MatrixXd R = Tridiag;

    std::cout << R << '\n';

    int row_num = Tridiag.rows();

    // Scale R 
    double max_val = 0, sum = 0.0, sigma, tau;;
    for (int i = 0; i < row_num; ++i){
        for (int j = i; j < row_num; ++j){
            if (R(i, j) > max_val){
                max_val = R(i,j);
            }
        }
    }
    bool sing;
    
    // Defining vectors for algorithm
    MatrixXd cdiag(row_num, 1), diag(row_num,1);
    std::cout << "max_val is: " << max_val << '\n';

    for (size_t i = 0; i < row_num; ++i){

        //std::cout << i << '\n';
        if (max_val == 0.0){
            sing = true;
            cdiag(i) = diag(i) = 0.0;
        }
        else{
            // I may have mixed up the indices...
            for (size_t j = i; j < row_num; ++j){
                //std::cout << "j = " << j  << '\n';
                R(i,j) = R(i,j) / max_val;
                sum += R(i,j) * R(i,j);
            }
            std::cout << "R check " << i << ": " << '\n' << R << '\n';
            sigma = sqrt(sum) * (double)sign(R(i,i));
            R(i,i) += sigma;
            cdiag(i) = sigma * R(i,i);
            diag(i) = -max_val * R(i,i); 
            for (size_t j = i+1; j < row_num; ++j){
                //std::cout << "j2 = " << j  << '\n';
                sum = 0.0;
                for (size_t k = i; k < row_num; k++){
                    sum += R(i, j) * R(k, j);
                }
                tau = sum / cdiag(i);
                std::cout << "tau is: " << tau << '\n';
                for (size_t k = i; k < row_num; k++){
                    //std::cout << "k = " << k << '\n';
                    R(i, j) -= tau * R(i,k);
                    
                }
            }
        }
    }

    std::cout << "finished scaling R" << '\n';
    std::cout << "sing is: " << sing << '\n';
    std::cout << "R is: " << '\n' << R << '\n';
    // the "-1" needs verification
    diag(row_num-1) = R(row_num-1, row_num-1);
    if (diag(row_num-1) == 0.0){
        sing = true;
    }

    std::cout << "checked diagonals" << '\n';

    // Set up Q explicitly
    for (int i = 0; i < row_num; ++i){
        if (cdiag(i) != 0.0){
            for (int j = 0; j < row_num; ++j){
                sum = 0.0;
                for (int k = i; k < row_num; ++k){
                    sum += R(i, k) * Q(i,j);
                }
                sum /= cdiag(i);
                for (int k = i; k < row_num; ++k){
                    Q(i,j) -= sum * R(i,k);
                }
            }
        }
    }

    // Remove lower left from R
    for (int i = 0; i < row_num; ++i){
        R(i,i) = diag(i);
        for (int k = 0; k < i; ++k){
            R(i,k) = 0.0;
        }
    }

    return Q;
}

// Function to return sign of value (signum function)
int sign(double value){
    if (value < 0.0){
        return -1;
    }
    else if (value > 0){
        return 1;
    }
    else {
        return 0;
    }
}

