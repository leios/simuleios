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

using namespace Eigen;

MatrixXd lanczos(MatrixXd d_matrix);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    MatrixXd a(2,2);
    a(0,0) = 4; a(1,0) = 1; a(0,1) = 1; a(1,1) = 3;

    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            std::cout << a(i,j) << '\t';
        }
    }
    std::cout << '\n';

    std::cout << a.rows() << '\n';

    MatrixXd Q = lanczos(a);

    for (size_t i = 0; i < Q.rows(); ++i){
        std::cout << Q(i) << '\n';
    }

    // generating identity matrix
    std::cout << MatrixXd::Identity(10,10) <<'\n';

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

MatrixXd lanczos(MatrixXd d_matrix){

    // Creating random device
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0,1); 

    double threshold = 0.01;
    int j = 0;
    int size = d_matrix.rows();

    // setting beta arbitrarily large for now 
    double beta = 10;

    // generating the first r 
    // NOTE: check 10 later
    MatrixXd r(d_matrix.rows(),1), q(d_matrix.rows(),1),
             a(d_matrix.rows(), d_matrix.cols());
    MatrixXd identity = MatrixXd::identity(d_matrix.rows(), d_matrix.cols());

    for (size_t i = 0; i < size; ++i){
        r(i) = dist(gen);
    }

    //while (beta > threshold){
    for (size_t i = 0; i < size; ++i){
        j = j + 1;
        beta = r.norm();
        q = r / beta;
        a = q.transpose() * d_matrix * q;
        r = (d_matrix - a * identity) * q - beta * q;
        std::cout << "i is: " << i << '\n';
    }
    return r;
}
