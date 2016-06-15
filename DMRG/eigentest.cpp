/*-------------eigentest.cpp--------------------------------------------------//
*
* Purpose: Simple multiplication to help visualize eigenvectors
*
*   Notes: compile with:
*            g++ -I /usr/include/eigen3/ eigentest.cpp -Wno-ignored-attributes -Wno-deprecated-declarations
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <Eigen/Core>
#include <vector>

using namespace Eigen;

int main(){

    int size = 10, count=0;

    // setting up positions of all the desired points to transform
    double x[size*size*size], y[size*size*size], z[size*size*size], 
           x2[size*size*size], y2[size*size*size], z2[size*size*size],
           xval = -1, yval = -1, zval = -1;
    MatrixXd Arr(3, 3);
    MatrixXd pos(3, 1);

    // Putting in "random" values for matrix
    Arr << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;
    std::cout << Arr << '\n';

    // Creating initial x, y, and z locations
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            for (int k = 0; k < size; ++k){
                xval = -1 + 2 * ((double)i / (double)size);
                yval = -1 + 2 * ((double)j / (double)size);
                zval = -1 + 2 * ((double)k / (double)size);
                x[count] = xval;
                y[count] = yval;
                z[count] = zval;

                // Performing multiplication / setting up vector
                pos(0) = xval;
                pos(1) = yval;
                pos(2) = zval;
                pos = Arr * pos;

                x2[count] = pos(0);
                y2[count] = pos(1);
                z2[count] = pos(2);
                std::cout << count << '\t' << xval << '\t' << yval << '\t' 
                          << zval << '\n';
                std::cout << '\n';
                std::cout << count << '\t' << x2[count] << '\t' 
                          << y2[count] << '\t' 
                          << z2[count] << '\n';
                count += 1;
            }
        }
    }

}
