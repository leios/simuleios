/*-------------FDTDTEz.cpp----------------------------------------------------//
*
*              Finite Difference Time Domain -- with TEz polarization
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

static const size_t spacey = 200;
static const size_t spacex = 400;
static const size_t losslayer = 20;

struct Bound{
    int x,y;
};

struct Loss{
    std::vector <double> ExH = std::vector<double>(spacex * spacey, 0), 
                         ExE = std::vector<double>(spacex * spacey, 0), 
                         EyE = std::vector<double>(spacex * spacey, 0), 
                         EyH = std::vector<double>(spacex * spacey, 0),
                         HzE = std::vector<double>(spacex * spacey, 0), 
                         HzH = std::vector<double>(spacex * spacey, 0);
};

struct Loss1d{
    std::vector <double> EyH = std::vector<double>(spacex, 0), 
                         EyE = std::vector<double>(spacex, 0), 
                         HzE = std::vector<double>(spacex, 0), 
                         HzH = std::vector<double>(spacex, 0);
};

struct Field{
    std::vector <double> Hz = std::vector<double>(spacex * spacey, 0), 
                         Ey = std::vector<double>(spacex * spacey, 0),
                         Ex = std::vector<double>(spacex * spacey, 0);

    std::vector <double> Hz1d = std::vector<double>(spacex + losslayer, 0), 
                         Ey1d = std::vector<double>(spacex + losslayer, 0);

    // 6 elements, 3 spacial elements away from border and 2 time elements of
    // those spatial elements
    std::vector <double> Etop = std::vector<double>(3 * 2 * spacex, 0), 
                         Ebot = std::vector<double>(3 * 2 * spacex, 0),
                         Eleft = std::vector<double>(3 * 2 * spacey, 0),
                         Eright = std::vector<double>(3 * 2 * spacey, 0);

    int t;
};

#define ExH(i, j) ExH[(i) + (j) *  spacex]
#define ExE(i, j) ExE[(i) + (j) *  spacex]
#define EyH(i, j) EyH[(i) + (j) *  spacex]
#define EyE(i, j) EyE[(i) + (j) *  spacex]
#define HzH(i, j) HzH[(i) + (j) *  spacex]
#define HzE(i, j) HzE[(i) + (j) *  spacex]
#define Hz(i, j) Hz[(i) + (j) *  spacex] 
#define Ey(i, j) Ey[(i) + (j) *  spacex] 
#define Ex(i, j) Ex[(i) + (j) *  spacex] 
#define Etop(k, j, i) Etop[(i) * 6 + (j) * 3 + (k)]
#define Ebot(k, j, i) Ebot[(i) * 6 + (j) * 3 + (k)]
#define Eleft(i, j, k) Eleft[(k) * 6 + (j) * 3 + (i)]
#define Eright(i, j, k) Eright[(k) * 6 + (j) * 3 + (i)]


void FDTD(Field EM,
          const int final_time, const double eps,
          std::ofstream& output);

// Adding ricker solutuion
double ricker(int time, int loc, double Cour);

// Adding plane wave
double planewave(int time, int loc, double Cour, int ppw, double radius);

// 2 dimensional functions for E / H movement
Field Hupdate2d(Field EM, Loss lass, int t);
Field Eupdate2d(Field EM, Loss lass, int t);

// 1 dimensional update functions for E / H
Field Hupdate1d(Field EM, Loss1d lass1d, int t);
Field Eupdate1d(Field EM, Loss1d lass1d, int t);

// Creating loss
Loss createloss2d(Loss lass, double eps, double Cour, double loss);
Loss1d createloss1d(Loss1d lass1d, double eps, double Cour, double loss);

// Total Field Scattered Field (TFSF) boundaries
Field TFSF(Field EM, Loss lass, Loss1d lass1d, double Cour);

// Checking Absorbing Boundary Conditions (ABS)
Field ABCcheck(Field EM, Loss lass);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int final_time = 2000;
    double eps = 377.0;

    Field EM;
    EM.t = 0;

    FDTD(EM, final_time, eps, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(Field EM,
          const int final_time, const double eps,
          std::ofstream& output){

    double loss = 0.00;
    double Cour = 1 / sqrt(2);

    Loss lass;
    lass  = createloss2d(lass, eps, Cour, loss);
    Loss1d lass1d;
    lass1d = createloss1d(lass1d, eps, Cour, loss);

    // Time looping
    for (int t = 0; t < final_time; t++){

        EM = Hupdate2d(EM, lass, t);
        EM = TFSF(EM, lass, lass1d, Cour);
        EM = Eupdate2d(EM,lass,t);
        EM = ABCcheck(EM, lass);
        // EM.Ey(200,100) = ricker(t, 0, Cour);
        
        // Outputting to a file
        int check = 5;
        if (t % check == 0){
            for (int dx = 0; dx < spacex; dx = dx++){
                for (int dy = 0; dy < spacey; dy = dy++){
                    output << t << '\t' << dx <<'\t' << dy << '\t'
                           << EM.Hz(dx, dy) << '\t' << EM.Ey(dx, dy) 
                           << '\t' << EM.Ex(dx, dy) << '\t' << '\n';
                }
            }

            output << '\n' << '\n';
        }

    }
}

// Adding the ricker solution
double ricker(int time, int loc, double Cour){
    double Ricky;
    double temp_const = 3.14159*(Cour*((double)time - (double)loc)/20.0 - 1.0);
    temp_const = temp_const * temp_const;
    Ricky = (1.0 - 2.0 * temp_const) * exp(-temp_const);
    return Ricky;

}

// 2 dimensional functions for E / H movement
Field Hupdate2d(Field EM, Loss lass, int t){
    // update magnetic field, x direction
    for (size_t dx = 0; dx < spacex - 1; dx++){
        for (size_t dy = 0; dy < spacey - 1; dy++){
           EM.Hz(dx,dy) = lass.HzH(dx,dy) * EM.Hz(dx, dy) 
                       + lass.HzE(dx,dy) * ((EM.Ex(dx, dy + 1)
                                         - EM.Ex(dx, dy))
                                         - (EM.Ey(dx + 1,dy)
                                         - EM.Ey(dx, dy)));
        }
    }

    return EM;

}

Field Eupdate2d(Field EM, Loss lass, int t){
    // update electric field
    for (size_t dx = 0; dx < spacex - 1; dx++){
        for (size_t dy = 1; dy < spacey - 1; dy++){
           EM.Ex(dx,dy) = lass.ExE(dx,dy) * EM.Ex(dx,dy)
                       + lass.ExH(dx,dy) * (EM.Hz(dx, dy) - EM.Hz(dx, dy - 1));
        }
    }

    for (size_t dx = 1; dx < spacex - 1; dx++){
        for (size_t dy = 0; dy < spacey - 1; dy++){
           EM.Ey(dx,dy) = lass.EyE(dx,dy) * EM.Ey(dx,dy)
                       - lass.EyH(dx,dy) * (EM.Hz(dx, dy) - EM.Hz(dx - 1, dy));
        }
    }

    return EM;
}

// 1 dimensional update functions for E / H
Field Hupdate1d(Field EM, Loss1d lass1d, int t){
    // update magnetic field, y direction
    for (size_t dx = 0; dx < spacex - 1; dx++){
        EM.Hz1d[dx] = lass1d.HzH[dx] * EM.Hz1d[dx] 
                  + lass1d.HzE[dx] * (EM.Hz1d[dx + 1] - EM.Hz1d[dx]);
    }

    return EM;
}

Field Eupdate1d(Field EM, Loss1d lass1d, int t){
    // update electric field, y direction
    for (size_t dx = 1; dx < spacex - 1; dx++){
        EM.Ey1d[dx] = lass1d.EyE[dx] * EM.Ey1d[dx] 
                  + lass1d.EyH[dx] * (EM.Hz1d[dx] - EM.Hz1d[dx - 1]);
    }

    return EM;

}

// Creating loss
Loss createloss2d(Loss lass, double eps, double Cour, double loss){

    double radius = 40;
    int sourcex = 200, sourcex2 = 100;
    int sourcey = 100, sourcey2 = 100;
    double dist, var, Q, epsp, mup, dist2;
    for (size_t dx = 0; dx < spacex; dx++){
        for (size_t dy = 0; dy < spacey; dy++){
             dist = sqrt((dx - sourcex)*(dx - sourcex) 
                       + (dy - sourcey)*(dy - sourcey)); 
             dist2 = sqrt((dx - sourcex2)*(dx - sourcex2) 
                        + (dy - sourcey2)*(dy - sourcey2)); 

            // if (dx > 100 && dx < 150 && dy > 75 && dy < 125){
            if (dist < radius){
                Q = cbrt(-(radius / dist) + sqrt((radius/dist) 
                                               * (radius/dist) + (1.0/27.0)));
                var = (Q - (1.0 / (3.0 * Q))) * (Q - (1.0/ (3.0 * Q)));
                // var = 1.4;
                if (abs(var) > 1000){
                    var = 1000;
                }

                if (isnan(var)){
                    var = 1000;
                }
                
                epsp = eps / (var * var);
                mup = 1 / (var * var);

                lass.ExH(dx, dy) = Cour * epsp /(1.0 - loss);
                lass.ExE(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.EyE(dx, dy) = (1.0 - loss) / (1.0 + loss);
                lass.EyH(dx, dy) = Cour * epsp / (1.0 + loss);
                lass.HzE(dx, dy) = Cour * (mup / eps) / (1.0 + loss);
                lass.HzH(dx, dy) = (1.0 - loss) / (1.0 + loss);

/*
                // PEC stuff -- not complete!
                lass.ExH(dx, dy) = 0;
                lass.ExE(dx, dy) = 0;
                lass.HzH(dx, dy) = 0;
                lass.HzE(dx, dy) = 0;
                lass.ExE(dx, dy) = 0;
                lass.ExH(dx, dy) = 0;
*/

            }
            else{

                lass.ExH(dx, dy) = Cour * eps;
                lass.ExE(dx, dy) = 1.0;
                lass.EyE(dx, dy) = 1.0;
                lass.EyH(dx, dy) = Cour * eps;
                lass.HzE(dx, dy) = Cour / eps;
                lass.HzH(dx, dy) = 1.0;
                
            }
        }
    }


    return lass;
}
Loss1d createloss1d(Loss1d lass1d, double eps, double Cour, double loss){

    double depth, lossfactor;

    for (size_t dx = 0; dx < spacex; dx++){
        if (dx < spacex - 1 - losslayer){
            lass1d.EyH[dx] = Cour * eps;
            lass1d.EyE[dx] = 1.0;
            lass1d.HzH[dx] = 1.0;
            lass1d.HzE[dx] = Cour / eps;
        }
        else{
            depth = dx - spacex - 1 - losslayer - 0.5;
            lossfactor = loss * pow(depth / (double)losslayer, 2);
            lass1d.EyH[dx] = Cour * eps / (1.0 + lossfactor);
            lass1d.EyE[dx] = (1.0 - lossfactor) / (1.0 + lossfactor);
            depth += 0.5;
            lass1d.HzH[dx] = (1.0 - lossfactor) / (1.0 + lossfactor);
            lass1d.HzE[dx] = Cour / eps / (1.0 + lossfactor);

        }
    }


    return lass1d;

}

// TFSF boundaries
Field TFSF(Field EM, Loss lass, Loss1d lass1d, double Cour){

    int dx, dy;

    // TFSF boundary
    Bound first, last;
    first.x = 10; last.x = 390;
    first.y = 10; last.y = 190;

    // Update along right
    dx = last.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Hz(dx, dy) -= lass.HzE(dx, dy) * EM.Ey1d[dx];
    }

    // Updating Hz along left
    dx = first.x - 1;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Hz(dx, dy) += lass.HzE(dx, dy) * EM.Ey1d[dx + 1];
    }

    // Insert 1d grid stuff here. Update magnetic and electric field
    Hupdate1d(EM, lass1d, EM.t);
    Eupdate1d(EM, lass1d, EM.t);
    //EM.Ey1d[10] = ricker(EM.t,0, Cour);
    EM.Ey1d[10] = planewave(EM.t, 15, Cour, 30, 40);
    EM.t++;
    std::cout << EM.t << '\n';

    // Check mag instead of ricker.
    // Update along right edge!
    dx = last.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Ey(dx,dy) -= lass.EyH(dx, dy) * EM.Hz1d[dx];
    }

    // Updating along left edge
    dx = first.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Ey(dx,dy) += lass.EyH(dx, dy) * EM.Hz1d[dx-1];
    }

    // Updating along top
    dy = last.y;
    for (int dx = first.x; dx <= last.x; dx++){
        EM.Ex(dx,dy) += lass.ExH(dx, dy) * EM.Hz1d[dx];
    }

    // Update along bot
    dy = first.y;
    for (int dx = first.x; dx <= last.x; dx++){
        EM.Ex(dx,dy) -= lass.ExH(dx, dy) * EM.Hz1d[dx];
    }

    return EM;

}

// Checking Absorbing Boundary Conditions (ABC)
Field ABCcheck(Field EM, Loss lass){

    // defining constant for  ABC
    double c1, c2, c3, temp1, temp2;
    temp1 = sqrt(lass.ExH(0,0) * lass.HzE(0,0));
    temp2 = 1.0 / temp1 + 2.0 + temp1;
    c1 = -(1.0 / temp1 - 2.0 + temp1) / temp2;
    c2 = -2.0 * (temp1 - 1.0 / temp1) / temp2;
    c3 = 4.0 * (temp1 + 1.0 / temp1) / temp2;
    size_t dx, dy;

    // Setting ABC for top
    for (dx = 0; dx < spacex; dx++){
        EM.Ex(dx, spacey - 1) = c1 * (EM.Ex(dx, spacey - 3) + EM.Etop(0, 1, dx))
                      + c2 * (EM.Etop(0, 0, dx) + EM.Etop(2, 0 , dx)
                              -EM.Ex(dx,spacey - 2) -EM.Etop(1, 1, dx))
                      + c3 * EM.Etop(1, 0, dx) - EM.Etop(2, 1, dx); 

       // memorizing fields...
        for (dy = 0; dy < 3; dy++){
            EM.Etop(dy, 1, dx) = EM.Etop(dy, 0, dx);
            EM.Etop(dy, 0, dx) = EM.Ex(dx, spacey - 1 - dy);
        }
    }

    // Setting ABC for bottom
    for (dx = 0; dx < spacex; dx++){
        EM.Ex(dx,0) = c1 * (EM.Ex(dx, 2) + EM.Ebot(0, 1, dx))
                      + c2 * (EM.Ebot(0, 0, dx) + EM.Ebot(2, 0 , dx)
                              -EM.Ex(dx,1) -EM.Ebot(1, 1, dx))
                      + c3 * EM.Ebot(1, 0, dx) - EM.Ebot(2, 1, dx); 

        // memorizing fields...
        for (dy = 0; dy < 3; dy++){
            EM.Ebot(dy, 1, dx) = EM.Ebot(dy, 0, dx);
            EM.Ebot(dy, 0, dx) = EM.Ex(dx, dy);
        }
    }

    // ABC on right
    for (dy = 0; dy < spacey; dy++){
        EM.Ey(spacex - 1,dy) = c1 * (EM.Ey(spacex - 3,dy) + EM.Eright(0, 1, dy))
                      + c2 * (EM.Eright(0, 0, dy) + EM.Eright(2, 0 , dy)
                              -EM.Ey(spacex - 2,dy) -EM.Eright(1, 1, dy))
                      + c3 * EM.Eright(1, 0, dy) - EM.Eright(2, 1, dy); 

        // memorizing fields...
        for (dx = 0; dx < 3; dx++){
            EM.Eright(dx, 1, dy) = EM.Eright(dx, 0, dy);
            EM.Eright(dx, 0, dy) = EM.Ey(spacex - 1 - dx, dy);
        }
    }


    // Setting ABC for left side of grid. Woo!
    for (dy = 0; dy < spacey; dy++){
        EM.Ey(0,dy) = c1 * (EM.Ey(2,dy) + EM.Eleft(0, 1, dy))
                      + c2 * (EM.Eleft(0, 0, dy) + EM.Eleft(2, 0 , dy)
                              -EM.Ey(1,dy) -EM.Eleft(1, 1, dy))
                      + c3 * EM.Eleft(1, 0, dy) - EM.Eleft(2, 1, dy); 

        // memorizing fields...
        for (dx = 0; dx < 3; dx++){
            EM.Eleft(dx, 1, dy) = EM.Eleft(dx, 0, dy);
            EM.Eleft(dx, 0, dy) = EM.Ey(dx, dy);
        }
    }

    return EM;
}


// Adding plane wave
double planewave(int time, int loc, double Cour, int ppw, double radius){
    double plane;

    plane = sin((2 * M_PI / (double)ppw) * (Cour * (double)time -
                 (double)loc));
    //plane = sin((double)(time-loc) * 3.5 / radius);

    return plane;
}

