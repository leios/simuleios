/*-------------FDTD.cpp-------------------------------------------------------//
*
*              Finite Difference Time Domain
*
* Purpose: To replicate the results of our invisible lense raytracer with 
*          FDTD. Woo!
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// This is the function we writs the bulk of the code in
void FDTD(std::vector<double>& Ez, std::vector<double>& Hy,
          const int final_time, const double eps, std::ofstream& output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int space = 200, final_time = 200;
    double eps = 377.0;

    // define initial E and H fields
    std::vector<double> Ez(space, 0.0), Hy(space, 0.0);

    FDTD(Ez, Hy, final_time, eps, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(std::vector<double>& Ez, std::vector<double>& Hy, 
          const int final_time, const double eps, std::ofstream& output){

    int space = Hy.size();

    // Time looping
    for (size_t t = 0; t < final_time; t++){

        // update magnetic field
        for (size_t dx = 0 ; dx < space - 1; dx++){
            Hy[dx] += (Ez[dx + 1] - Ez[dx]) / eps;
        }

        // update electric field
        for (size_t dx = 1; dx < space; dx++){
            Ez[dx] += (Hy[dx] - Hy[dx-1]) / eps;
        }

        // set src for next step
        Ez[0] = exp(-(t - 30.0) * (t - 30.0)/100.0);

        output << Ez[50] << '\n';

    }
}
