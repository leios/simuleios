/*-------------determinant.cpp------------------------------------------------//
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
#include <fstream>

using namespace Eigen;

int main(){

    // Opening file to writing
    std::ofstream output;
    output.open("out.dat");

    int size = 2, count=0;

    // setting up positions of all the desired points to transform
    double x[size*size*size], y[size*size*size], z[size*size*size], 
           x2[size*size*size], y2[size*size*size], z2[size*size*size],
           xval = -1, yval = -1, zval = -1;
    MatrixXd Arr(3, 3);
    MatrixXd pos(3, 1);

    // Putting in "random" values for matrix
    Arr << 1, 2, 0,
           2, 1, 0,
           0, 0, -3;
/*
    Arr << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;
*/
    std::cout << Arr << '\n';

    // Creating initial x, y, and z locations
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            for (int k = 0; k < size; ++k){
                xval = -1 + 2 * ((double)i / ((double)size - 1));
                yval = -1 + 2 * ((double)j / ((double)size - 1));
                zval = -1 + 2 * ((double)k / ((double)size - 1));
                x[count] = xval;
                y[count] = yval;
                z[count] = zval;

                // Performing multiplication / setting up vector
                pos(0) = xval;
                pos(1) = yval;
                pos(2) = zval;
                pos = Arr * pos;

                // Storing values in x2, y2, and z2
                x2[count] = pos(0);
                y2[count] = pos(1);
                z2[count] = pos(2);
                count += 1;
            }
        }
    }
    count = 8;
    for (int i = 0; i < count; ++i){
        std::cout << x[i] << '\t' << y[i] << '\t' << z[i] << '\n';
        pos(0) = x[i];
        pos(1) = y[i];
        pos(2) = z[i];

        pos = Arr * pos;

        // Storing values in x2, y2, and z2
        x2[i] = pos(0);
        y2[i] = pos(1);
        z2[i] = pos(2);

    }
/*

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

                // Storing values in x2, y2, and z2
                x2[count] = pos(0);
                y2[count] = pos(1);
                z2[count] = pos(2);
                count += 1;
            }
        }
    }


    // Writing to file in correct format
    count = 0;
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            for (int k = 0; k < size; ++k){
                output << x[count] << '\t' << y[count]
                       << '\t' << z[count] << '\t' << count << '\n';
                count++;
            }
        }
    } 

    output << '\n' << '\n';

    count = 0;
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            for (int k = 0; k < size; ++k){
                output << x2[count] << '\t' << y2[count]
                       << '\t' << z2[count] << '\t' << count << '\n';
                count++;
            }
        }
    } 
*/

    int frames = 60;
    count = 0;
    double xvel[size*size*size], yvel[size*size*size], zvel[size*size*size];
    double vid_time = 1.0;

    count = 8;

    for (int i = 0; i < count; ++i){
        xvel[i] = (x[i] - x2[i]) / vid_time;
        yvel[i] = (y[i] - y2[i]) / vid_time;
        zvel[i] = (z[i] - z2[i]) / vid_time;
    }

    for (int f = 0; f < frames; ++f){
        for (int i = 0; i < count; ++i){
            xval = x[i] + ((x2[i] - x[i])
                   * ((double)f / (double)frames));
            yval = y[i] + ((y2[i] - y[i])
                   * ((double)f / (double)frames));
            zval = z[i] + ((z2[i] - z[i])
                   * ((double)f / (double)frames));

            output << xval << '\t' << yval
                   << '\t' << zval << '\t'
                   << xvel[i] << '\t' << yvel[i] << '\t'
                   << zvel[i] << '\t' << 1 << '\t' << i << '\n';

        }
        output << '\n' << '\n';
    }

    /*
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            for (int k = 0; k < size; ++k){
                xvel[count] = (x[count] - x2[count]) / vid_time;
                yvel[count] = (y[count] - y2[count]) / vid_time;
                zvel[count] = (z[count] - z2[count]) / vid_time;
                count = count + 1;
            }
        }
    }

    for (int f = 0; f < frames; ++f){
        count = 0;
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                for (int k = 0; k < size; ++k){
                    xval = x[count] + ((x2[count] - x[count])
                           * ((double)f / (double)frames));
                    yval = y[count] + ((y2[count] - y[count])
                           * ((double)f / (double)frames));
                    zval = z[count] + ((z2[count] - z[count])
                           * ((double)f / (double)frames));

                    output << xval << '\t' << yval
                           << '\t' << zval << '\t'
                           << xvel[count] << '\t' << yvel[count] << '\t'
                           << zvel[count] << '\t' << 1 << '\t' << count << '\n';
                    count++;
                }
            }
        }

        output << '\n' << '\n';
    }
    */

    output.close();

}
