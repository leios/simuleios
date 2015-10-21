/*-------------FDTD.cpp-------------------------------------------------------//
*
*              Finite Difference Time Domain
*
* Purpose: To replicate the results of our invisible lense raytracer with 
*          FDTD. Woo!
*
*   Notes: Most of this is coming from the following link:
*             http://www.eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// This is the function we writes the bulk of the code in
void FDTD(std::vector<double>& Ez, std::vector<double>& Hy,
          const int final_time, const double eps, const int space, 
          std::ofstream& output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int space = 200, final_time = 500;
    double eps = 377.0;

    // define initial E and H fields
    std::vector<double> Ez(space, 0.0), Hy(space, 0.0);

    FDTD(Ez, Hy, final_time, eps, space, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(std::vector<double>& Ez, std::vector<double>& Hy, 
          const int final_time, const double eps, const int space,
          std::ofstream& output){

    // For magnetic field:
    // double offset = 0.00005;

    // for electric field:
    double offset = 0.05;
    double loss = 0.01;

    // Relative permittivity
    std::vector<double> LossEzH(space, 0), LossEzE(space, 0);
    std::vector<double> LossHyH(space, 0), LossHyE(space, 0);
    for (int dx= 0; dx < space; dx++){
        if (dx > 100 && dx < 150){
            LossEzH[dx] = eps / 9.0 /(1.0 - loss);
            LossEzE[dx] = (1.0 - loss) / (1.0 + loss);
            LossHyH[dx] = (1.0 - loss) / (1.0 + loss);
            LossHyE[dx] = (1.0 / eps) / (1.0 + loss);
        }
        else{
            LossEzH[dx] = eps;
            LossEzE[dx] = 1.0;
            LossHyH[dx] = 1.0;
            LossHyE[dx] = (1.0 / eps);
        }
    }

    // Time looping
    for (int t = 0; t < final_time; t++){

        // Linking the final two elements for an ABC
        // Hy[space - 1] = Hy[space - 2];

        // update magnetic field
        for (int dx = 0; dx < space - 1; dx++){
            Hy[dx] = LossHyH[dx] * Hy[dx] + LossHyE[dx] * (Ez[dx + 1] - Ez[dx]);

        }

        // Correction to the H field for the TFSF boundary
        Hy[49] -= exp(-(t - 40.) * (t - 40.)/100.0) / eps;
        // Hy[49] -= sin((t-10.0)*0.2)*0.0005;


        // Linking the first two elements in the electric field
        Ez[0] = Ez[1];
        Ez[space - 1] = Ez[space - 2];

        // update electric field
        for (int dx = 1; dx < space - 1; dx++){
            Ez[dx] = LossEzE[dx] * Ez[dx] + 
                     LossEzH[dx] * (Hy[dx] - Hy[dx-1]);
        }

        // set src for next step
        Ez[50] += exp(-((t +1 - 40.) * (t + 1 - 40.))/100.0);
        
        // Ez[50] += sin((t-10.0)*0.2)*0.0005;
/*
        if (t > 0){
            Ez[50] = sin(0.1 * t) / (0.1 * t);
        }
        else{
            Ez[50] = 1;
        }
*/
        if (t % 10 == 0){
            for (int dx = 0; dx < space; dx++){
                output << Ez[dx] + (t * offset) << '\n';
                //output << Hy[dx] + (t * offset) << '\n';
            }

            output << '\n' << '\n';
        }

    }
}
