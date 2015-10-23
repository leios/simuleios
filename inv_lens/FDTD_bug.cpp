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

static const size_t SPACE = 200;

struct Mat : std::vector<double> {
    using std::vector<double>::vector;

    double operator()(size_t i, size_t j) const { return (*this)[i + j * SPACE]; }
    double& operator()(size_t i, size_t j) { return (*this)[j + j * SPACE]; }
};

struct Loss {
    Mat EzH = {SPACE * SPACE, 0};
    Mat EzE = {SPACE * SPACE, 0};
    Mat HyE = {SPACE * SPACE, 0};
    Mat HyH = {SPACE * SPACE, 0};
    Mat HxE = {SPACE * SPACE, 0};
    Mat HxH = {SPACE * SPACE, 0};
};

struct Field {
    Mat Hx = {SPACE * SPACE, 0};
    Mat Hy = {SPACE * SPACE, 0};
    Mat Ez = {SPACE * SPACE, 0};
};

void FDTD(std::vector<double>& Ez, std::vector<double>& Hy,
          const int final_time, const double eps, const int space, Loss lass,
          std::ofstream& output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int space = 200, final_time = 500;
    double eps = 377.0;

    Loss lass;

    // define initial E and H fields
    std::vector<double> Ez(SPACE, 0.0), Hy(SPACE, 0.0);

    FDTD(Ez, Hy, final_time, eps, space, lass, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(std::vector<double>& Ez, std::vector<double>& Hy,
          const int final_time, const double eps, const int space, Loss lass,
          std::ofstream& output){

    Field field;
    // For magnetic field:
    // double offset = 0.00005;

    // for electric field:
    double offset = 0.05;
    double loss = 0.0;

    // Relative permittivity
    for (int dx = 0; dx < space; dx++){
        for (int dy = 0; dy < space; dy++){
            if (dx > 100 && dx < 150){
                lass.EzH(dx, dy) =  eps / 9.0 /(1.0 - loss);
                lass.EzE(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.HyH(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.HyE(dx, dy) = (1.0 / eps) / (1.0 + loss);
                lass.HxE(dx, dy) = (1.0 / eps) / (1.0 + loss);
                lass.HxH(dx, dy) = (1.0 - loss) / (1.0 + loss);
            }
            else{
                lass.EzH(dx, dy) =  eps;
                lass.EzE(dx, dy) = 1.0;
                lass.HyH(dx, dy) = 1.0;
                lass.HyE(dx, dy) = (1.0 / eps);
                lass.HxE(dx, dy) = (1.0 / eps);
                lass.HxH(dx, dy) = 1.0;
            }
        }
    }

    // Time looping
    for (int t = 0; t < final_time; t++){

        // Linking the final two elements for an ABC
        // Hy[space - 1] = Hy[space - 2];

        // update magnetic field, y direction
        for (int dx = 0; dx < space - 1; dx++){
            for (int dy = 0; dy < space; dy++){
                field.Hy(dx,dy) = lass.HyH(dx,dy) * field.Hy(dx,dy)
                           + lass.HyE(dx,dy) * (field.Ez(dx + 1,dy)
                                                - field.Ez(dx,dy));
            }
        }

        // update magnetic field, x direction
        for (int dx = 0; dx < space; dx++){
            for (int dy = 0; dy < space - 1; dy++){
                field.Hx(dx,dy) = lass.HxH(dx,dy) * field.Hx(dx, dy)
                           + lass.HxE(dx,dy) * (field.Ez(dx + 1,dy)
                                                - field.Ez(dx,dy));
            }
        }


        // Correction to the H field for the TFSF boundary
        // Hy[49] -= exp(-(t - 40.) * (t - 40.)/100.0) / eps;
        // Hy[49] -= sin((t-10.0)*0.2)*0.0005;


        // Linking the first two elements in the electric field
        field.Ez[0] = field.Ez[1];
        // Ez[space - 1] = Ez[space - 2];

        // update electric field
        for (int dx = 0; dx < space - 1; dx++){
            for (int dy = 0; dy < space - 1; dy++){
               field.Ez(dx,dy) = lass.EzE(dx,dy) * field.Ez(dx,dy)
                           + lass.EzH[dx] * (field.Hy(dx, dy)
                                             - field.Hy(dx-1, dy)
                                             - field.Hx(dx,dy)
                                             - field.Hx(dx, dy - 1));
            }
        }

        // set src for next step
        Ez[0] += exp(-((t + 1 - 40.) * (t + 1 - 40.))/100.0);

        // Ez[50] += sin((t - 10.0 + 1)*0.2)*0.0005;
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
                for (int dy = 0; dy < space; dy++){
                    output << t << field.Ez(dx, dy) << '\n';
                    //output << Ez(dx,dy) + (t * offset) << '\n';
                    //output << Hy[dx] + (t * offset) << '\n';
                }
            }

            output << '\n' << '\n';
        }

    }
}
