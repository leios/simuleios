/*-------------FDTD.cpp-------------------------------------------------------//
*
*              Finite Difference Time Domain
*
* Purpose: To replicate the results of our invisible lense raytracer with 
*          FDTD. Woo!
*
*   Notes: Most of this is coming from the following link:
*             http://www.eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf
*             http://www.eecs.wsu.edu/~schneidj/ufdtd/chap8.pdf
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

static const size_t space = 200;

struct Bound{
    int x,y;
};

struct Loss{
    std::vector <double> EzH = std::vector<double>(space * space, 0), 
                         EzE = std::vector<double>(space * space, 0), 
                         HyE = std::vector<double>(space * space, 0), 
                         HyH = std::vector<double>(space * space, 0),
                         HxE = std::vector<double>(space * space, 0), 
                         HxH = std::vector<double>(space * space, 0);
};

struct Field{
    std::vector <double> Hx = std::vector<double>(space * space, 0), 
                         Hy = std::vector<double>(space * space, 0),
                         Ez = std::vector<double>(space * space, 0);
};

#define EzH(i, j) EzH[(i) + (j) *  space]
#define EzE(i, j) EzE[(i) + (j) *  space]
#define HyH(i, j) HyH[(i) + (j) *  space]
#define HyE(i, j) HyE[(i) + (j) *  space]
#define HxH(i, j) HxH[(i) + (j) *  space]
#define HxE(i, j) HxE[(i) + (j) *  space]
#define Hx(i, j) Hx[(i) + (j) *  space] 
#define Hy(i, j) Hy[(i) + (j) *  space] 
#define Ez(i, j) Ez[(i) + (j) *  space] 

void FDTD(Field EM,
          const int final_time, const double eps, const int space, Loss lass,
          std::ofstream& output);

// Adding ricker solutuion
double ricker(int time, int loc, double Cour);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int final_time = 1000;
    double eps = 377.0;

    Loss lass;

    // define initial E and H fields
    // std::vector<double> Ez(space, 0.0), Hy(space, 0.0);
    Field EM;

    FDTD(EM, final_time, eps, space, lass, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(Field EM,
          const int final_time, const double eps, const int space, Loss lass,
          std::ofstream& output){

    // For magnetic field:
    // double offset = 0.00005;

    // for electric field:
    //double offset = 0.05;
    double loss = 0.0;
    double Cour = 1.0 / sqrt(2.0);

    // Relative permittivity
    for (int dx = 0; dx < space; dx++){
        for (int dy = 0; dy < space; dy++){
            //if (dx > 100 && dx < 150){
/*
                lass.EzH(dx, dy) = Cour * eps;
                lass.EzE(dx, dy) = 1.0;
                lass.HyH(dx, dy) = 1.0;
                lass.HyE(dx, dy) = Cour / eps;
                lass.HxE(dx, dy) = Cour / eps;
                lass.HxH(dx, dy) = 1.0;
*/

                lass.EzH(dx, dy) =  eps / 9.0 /(1.0 - loss);
                lass.EzE(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.HyH(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.HyE(dx, dy) = (1.0 / eps) / (1.0 + loss);
                lass.HxE(dx, dy) = (1.0 / eps) / (1.0 + loss);
                lass.HxH(dx, dy) = (1.0 - loss) / (1.0 + loss);
            //}
            /*
            else{
                lass.EzH(dx, dy) =  Cour * eps;
                lass.EzE(dx, dy) = 1.0;
                lass.HyH(dx, dy) = 1.0;
                lass.HyE(dx, dy) = Cour / eps;
                lass.HxE(dx, dy) = Cour / eps;
                lass.HxH(dx, dy) = 1.0;
                
                lass.EzH(dx, dy) =  eps;
                lass.EzE(dx, dy) = 1.0;
                lass.HyH(dx, dy) = 1.0;
                lass.HyE(dx, dy) = (1.0 / eps);
                lass.HxE(dx, dy) = (1.0 / eps);
                lass.HxH(dx, dy) = 1.0;
                
            }
            */
        }
    }

    // Time looping
    for (int t = 0; t < final_time; t++){

        // update magnetic field, x direction
        for (int dx = 0; dx < space; dx++){
            for (int dy = 0; dy < space - 1; dy++){
               EM.Hx(dx,dy) = lass.HxH(dx,dy) * EM.Hx(dx, dy) 
                           - lass.HxE(dx,dy) * (EM.Ez(dx,dy + 1) 
                                                - EM.Ez(dx,dy));
            }
        }


        // update magnetic field, y direction
        for (int dx = 0; dx < space - 1; dx++){
            for (int dy = 0; dy < space; dy++){
               EM.Hy(dx,dy) = lass.HyH(dx,dy) * EM.Hy(dx,dy) 
                           + lass.HyE(dx,dy) * (EM.Ez(dx + 1,dy) 
                                                - EM.Ez(dx,dy));
            }
        }

        // TFSF boundary
        Bound first, last;
        first.x = 75; last.x = 125;
        first.y = 75; last.y = 125;

        // Updating along left edge
        int dx = first.x - 1;
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Hy(dx,dy) -= lass.HyE(dx, dy) * ricker(t,dx + 1, Cour);
        }

        // Update along right edge!
        dx = last.x;
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Hy(dx,dy) += lass.HyE(dx, dy) * ricker(t, dx, Cour);
        }

        // Update along bot
        int dy = first.y - 1;
        for (int dx = first.x; dx <= last.x; dx++){
            EM.Hx(dx,dy) += lass.HxE(dx, dy) * ricker(t, dx, Cour);
        }

        // Updating along top
        dy = last.y;
        for (int dx = first.x; dx <= last.x; dx++){
            EM.Hx(dx,dy) += lass.HxE(dx, dy) * ricker(t, dx, Cour);
        }

        // update electric field
        for (int dx = 1; dx < space - 1; dx++){
            for (int dy = 1; dy < space - 1; dy++){
               EM.Ez(dx,dy) = lass.EzE(dx,dy) * EM.Ez(dx,dy)
                           + lass.EzH(dx,dy) * ((EM.Hy(dx, dy)
                                             - EM.Hy(dx - 1, dy))
                                             - (EM.Hx(dx,dy)
                                             - EM.Hx(dx, dy - 1)));
            }
        }

        // Check mag instead of ricker.
        // Updating Ez along left
        dx = first.x;
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Ez(dx, dy) -= lass.EzH(dx, dy) * ricker(t, dx, Cour);
        }

        // Update along right
        dx = last.x;
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Ez(dx, dy) += lass.EzH(dx, dy) * ricker(t, dx - 1, Cour);
        }


        // set up the Ricker Solution in text -- src for next step
        /*
        double temp_const = 3.14159 * ((Cour * t - 0.0) / 20.0 - 1.0);
        temp_const = temp_const * temp_const;
        EM.Ez(100,100) = (1.0 - 2.0 * temp_const) * exp(-temp_const);
        */
        EM.Ez(100,100) = ricker(t, 0, Cour);

        
        // Outputting to a file
        int check = 5;
        if (t % check == 0){
            for (int dx = 0; dx < space; dx = dx + check){
                for (int dy = 0; dy < space; dy = dy + check){
                    output << t << '\t' << dx <<'\t' << dy << '\t'
                           << EM.Ez(dx, dy) << '\t' << EM.Hy(dx, dy) 
                           << '\t' << EM.Hx(dx, dy) << '\t' << '\n';
                    //output << Ez(dx,dy) + (t * offset) << '\n';
                    //output << Hy[dx] + (t * offset) << '\n';
                }
            }

            output << '\n' << '\n';
        }

    }
}

// Adding the ricker solution
double ricker(int time, int loc, double Cour){
    double Ricky;
    double temp_const = 3.14159*((Cour*(double)time - (double)loc)/20.0 - 1.0);
    temp_const = temp_const * temp_const;
    Ricky = (1.0 - 2.0 * temp_const) * exp(-temp_const);
    return Ricky;

}
