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
#include <Eigen/QR>
#include <random>
#include <vector>
#include <math.h>

using namespace Eigen;

// Function for the lanczos algorithm, returns Tri-diagonal matrix
MatrixXd lanczos(MatrixXd &d_matrix);

// Function for QR decomposition
MatrixXd qrdecomp(MatrixXd &Tridiag);

// Function to perform the Power Method
void p_method(MatrixXd &Tridiag, MatrixXd &Q);

// Function to return sign of value (signum function)
int sign(double value);

// Function to check eigenvectors and values
void eigentest(MatrixXd &d_matrix, MatrixXd &Q);

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

    MatrixXd Qtemp = Q;

    std::cout << Q << '\n';

    std::cout << "Finding eigenvalues: " << '\n';
    p_method(Tridiag, Q);

    Qtemp = Qtemp - Q;
    std::cout << "After the Power Method: " << Qtemp.squaredNorm() << '\n';
    eigentest(Tridiag, Q);

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function for the lanczos algorithm, returns Tri-diagonal matrix
MatrixXd lanczos(MatrixXd &d_matrix){

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
        // std::cout << j << '\n';
    }

    MatrixXd krylov_id = krylov.transpose() * krylov;
    std::cout << "The identity matrix from the krylov subspace is: \n" 
              << krylov_id << '\n';

    MatrixXd T(j_tot,j_tot);
    T = krylov.transpose() * d_matrix * krylov;

    return T;
}

// Function for QR decomposition
// Because we only need Q for the power method, I will retun only Q
MatrixXd qrdecomp(MatrixXd &Tridiag){
    // Q is and orthonormal vector => Q'Q = 1
    MatrixXd Q(Tridiag.rows(), Tridiag.cols());
    MatrixXd Id = MatrixXd::Identity(Tridiag.rows(), Tridiag.cols());

    // R is the upper triangular matrix
    MatrixXd R = Tridiag;

    int row_num = Tridiag.rows();
    int countx = 0, county = 0;

    // Scale R 
    double sum = 0.0, sigma, tau, fak, max_val = 0;

/*
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
*/

    bool sing;

    // Defining vectors for algorithm
    MatrixXd diag(row_num,1);

    for (size_t i = 0; i < row_num; ++i){

        // determining l_2 norm
        sum = 0.0;
        for (size_t j = i; j < row_num; ++j){
            sum += R(j,i) * R(j,i);
            std::cout << R(j,i) << '\n';
        }
        sum = sqrt(sum);

        std::cout << "sum is: " << sum << '\n';

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
            R(i,i) = R(i,i) - diag(i);
            for (size_t j = i; j < row_num; ++j){
                R(j,i) = R(j,i) / fak;
            }

            // Creating blocks to work with
            MatrixXd block1 = R.block(i, i+1, row_num-i, row_num -i-1);
            MatrixXd block2 = R.block(i, i, row_num-i,1);

            block1 = block1 - block2 * (block2.transpose() * block1);

            std::cout << R << '\n' <<  '\n' << block1 << '\n';
            // setting values back to what they need to be
            countx = 0;
            for (int j = i+1; j < row_num; ++j){
                for (int k = i; k < row_num; ++k){
                    R(k,j) = block1(k-i, j-i-1);
                }
            }
        }
    }


    MatrixXd z(row_num, 1);

    // Explicitly defining Q
    // Create column block for multiplication
    for (size_t i = 0; i < row_num; ++i){
        MatrixXd Idblock = Id.block(0, i, row_num, 1);
        for (int j = row_num-1; j >= 0; --j){
            z = Idblock;

            // Creating blocks for multiplication
            MatrixXd zblock = z.block(j, 0, row_num - j, 1);
            MatrixXd Rblock = R.block(j, j, row_num - j, 1);

            // Performing multiplication
            zblock = zblock - Rblock * (Rblock.transpose() * zblock);

            // Set xblock up for next iteration of k
            for (int k = j; k < row_num; ++k){
                z(k) = zblock(k-j); 
            }
        }

        Q.col(i) = z;

        //std::cout << Q << '\n';
    }

    // Remove lower left from R
    for (int i = 0; i < row_num; ++i){
        R(i,i) = diag(i);
        for (int j = 0; j < i; ++j){
            R(i,j) = 0;
        }
    }

    //std::cout << "R is: " << '\n' << R << '\n';

    MatrixXd temp = Q.transpose() * Q;

/*
    std::cout << "Truncated Q^T * Q is:" << '\n';
    for (int i = 0; i < temp.rows(); ++i){
        for (int j = 0; j < temp.cols(); ++j){
            if (temp(i,j) < 0.00000000001){
                std::cout << 0 << '\t';
            }
            else{
                std::cout << temp(i,j) <<'\t';
            }
        }
        std::cout << '\n';
    }
    std::cout << '\n';
*/

    //std::cout << "Q^T * Q is: " << '\n' << Q * Q.transpose() << '\n' << '\n';
    //std::cout << "QR - A is: " << '\n' << Q*R - Tridiag << '\n';
    //std::cout << "Q^T * A - R: " << '\n'
    //          << Q.transpose() * Tridiag - R << '\n' << '\n';

    return Q.transpose();
}

// Function to perform the Power Method
void p_method(MatrixXd &Tridiag, MatrixXd &Q){

    // Find all eigenvectors
    MatrixXd eigenvectors(Tridiag.rows(), Tridiag.cols());
    MatrixXd Z(Tridiag.rows(), Tridiag.cols());
    MatrixXd Qtemp = Q;

    // Iteratively defines eigenvectors
    for (int i = 0; i < Tridiag.rows(); ++i){
        Z = Tridiag * Q;
        Q = qrdecomp(Z);

    }

    Qtemp = Qtemp - Q;
    std::cout << "This should not be 0: " << Qtemp.squaredNorm() << '\n';

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

// Function to check eigenvectors and values
void eigentest(MatrixXd &Tridiag, MatrixXd &Q){

    // Calculating the Rayleigh quotient (v^t * A * v) / (v^t * v)
    // Note, that this should be a representation of eigenvalues

    std::vector<double> eigenvalues(Tridiag.rows());
    MatrixXd eigenvector(Tridiag.rows(),1);
    double QQ, QAQ;

    for (size_t i = 0; i < Tridiag.rows(); ++i){
        QQ = Q.col(i).transpose() * Q.col(i);    
        QAQ = Q.col(i).transpose() * Tridiag * Q.col(i);
        eigenvalues[i] =  QAQ / QQ;
        std::cout << "eigenvalue is: " << eigenvalues[i] << '\n';

        eigenvector = ((Tridiag * Q.col(i)) / eigenvalues[i]) - Q.col(i);
        std::cout << eigenvector << '\n' << '\n';
        std::cout << "This should be 0: " << '\t' 
                  << eigenvector.squaredNorm() << '\n';
        
    }

}

