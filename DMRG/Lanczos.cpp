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
        for (size_t j = 0; j <= i; ++j){
            d_matrix(i,j) = dist(gen);
            d_matrix(j,i) = d_matrix(i,j);
        }
    }

    MatrixXd Tridiag = lanczos(d_matrix);

    std::cout << '\n' << "Tridiagonal matrix is: \n";

    for (size_t i = 0; i < Tridiag.rows(); ++i){
        for (size_t j = 0; j < Tridiag.cols(); ++j){
            std::cout << Tridiag(i, j) << '\t';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

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

/*
    // Setting values to 0 if they are close...
    for (size_t i = 0; i < T.rows(); ++i){
        for (size_t j = 0; j < T.cols(); ++j){
            if (T(i,j) < 0.00001){
                T(i,j) = 0;
            }
        }
    }
*/

    return T;
}

// Function for QR decomposition
// Because we only need Q for the power method, I will retun only Q
MatrixXd qrdecomp(MatrixXd Tridiag){
    // Q is and orthonormal vector => Q'Q = 1
    MatrixXd Q(Tridiag.rows(), Tridiag.cols());
    MatrixXd Id = MatrixXd::Identity(Tridiag.rows(), Tridiag.cols());

    // R is the upper triangular matrix
    MatrixXd R = Tridiag;

    std::cout << R << '\n';

    int row_num = Tridiag.rows();
    int countx = 0, county = 0;

    std::cout << "row_num is: " << row_num << '\n';

    // Scale R 
    double sum = 0.0, sigma, tau, fak, max_val = 0;

    for (int i = 0; i < row_num; ++i){
        for (int j = 0; j < row_num; ++j){
            if (R(i,j) > max_val){
                max_val = R(i,j);
            }
        }
    }

    for (int i = 0; i < row_num; ++i){
        for (int j = 0; j < row_num; ++j){
            R(i,j) /= max_val;
        }
    }

    bool sing;

    // Defining vectors for algorithm
    MatrixXd diag(row_num,1);

    for (size_t i = 0; i < row_num; ++i){

        // determining l_2 norm
        sum = 0.0;
        for (size_t j = i; j < row_num; ++j){
            sum += R(j,i) * R(j,i);
        }
        sum = sqrt(sum);

        //std::cout << i << '\n';
        if (sum == 0.0){
            sing = true;
            diag(i) = 0.0;
            std::cout << "MATRIX IS SINGULAR!!!" << '\n';
        }
        else{

            if (R(i,i) >= 0){
                diag(i) = -sum;
            }
            else{
                diag(i) = sum;
            }
            fak = sqrt(sum * (sum + abs(R(i,i))));
            std::cout << "fak is: " << fak << '\n';
            R(i,i) = R(i,i) - diag(i);
            for (size_t j = i; j < row_num; ++j){
                R(j,i) = R(j,i) / fak;
            }

            // Creating blocks to work with
            MatrixXd block1 = R.block(i, i+1, row_num-i, row_num - i - 1);
            MatrixXd block2 = R.block(i, i, row_num-i,1);

            std::cout << "R is: " << '\n' << R << '\n';

            std::cout << "checking blocks:" << '\n';

            std::cout << block1 << '\n' << '\n' << block2 << '\n';

            block1 = block1 - block2 * (block2.transpose() * block1);

            // setting values back to what they need to be
            countx = 0;
            for (int j = i+1; j < row_num; ++j){
                county = 0;
                for (int k = i; k < row_num; ++k){
                    R(k,j) = block1(county, countx);
                    std::cout << k << '\t' << j << '\t' << countx << '\t' 
                              << county << '\t' << R(k,j) << '\n';
                    ++county;
                }
                ++countx;
            }
        }
    }

    std::cout << "R is: " << '\n';
    std::cout << R << '\n';

    MatrixXd z(row_num, 1);

    // Explicitly defining Q
    // Create column block for multiplication
    for (size_t i = 0; i < row_num; ++i){
        MatrixXd Idblock = Id.block(0, i, row_num, 1);
        std::cout << "i is: " << i << '\n';
        std::cout << "IDblock is: " << '\n' << Idblock << '\n';
        for (int j = row_num-1; j >= 0; --j){
            z = Idblock;

            std::cout << "j is: " << j << '\n';
            // Creating blocks for multiplication
            MatrixXd zblock = z.block(j, 0, row_num - j, 1);
            MatrixXd Rblock = R.block(j, j, row_num - j, 1);

            // Performing multiplication
            zblock = zblock - Rblock * (Rblock.transpose() * zblock);

            // Set xblock up for next iteration of k
            int count = 0;
            for (int k = j; k < row_num; ++k){
                z(k) = zblock(count); 
                ++count;
            }
        }

        std::cout << "got to here" << '\n';
        Q.col(i) = z;

        std::cout << Q << '\n';
    }

    // Remove lower left from R
    for (int i = 0; i < row_num; ++i){
        R(i,i) = diag(i);
        for (int j = 0; j < i; ++j){
            R(i,j) = 0;
        }
    }

    std::cout << "R is: " << '\n' << R << '\n';

    std::cout << "Q^T * Q is: " << '\n' << Q * Q.transpose() << '\n' << '\n';

    return Q.transpose();
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

