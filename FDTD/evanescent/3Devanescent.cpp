/*-------------3Devanescent.cpp-----------------------------------------------//
*
*              Finite Difference Time Domain -- evanescent field test
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

static const size_t spacey = 50;
static const size_t spacex = 50;
static const size_t spacez = 50;
static const size_t losslayer = 20;

struct Bound{
    int x,y,z;
};

struct Loss{
    std::vector <double> ExH = std::vector<double>(spacex * spacey * spacez, 0),
                         ExE = std::vector<double>(spacex * spacey * spacez, 0),
                         EyH = std::vector<double>(spacex * spacey * spacez, 0),
                         EyE = std::vector<double>(spacex * spacey * spacez, 0),
                         EzH = std::vector<double>(spacex * spacey * spacez, 0),
                         EzE = std::vector<double>(spacex * spacey * spacez, 0),

                         HxE = std::vector<double>(spacex * spacey * spacez, 0),
                         HxH = std::vector<double>(spacex * spacey * spacez, 0),
                         HyE = std::vector<double>(spacex * spacey * spacez, 0),
                         HyH = std::vector<double>(spacex * spacey * spacez, 0),
                         HzE = std::vector<double>(spacex * spacey * spacez, 0),
                         HzH = std::vector<double>(spacex * spacey * spacez, 0);
};

struct Loss1d{
    std::vector <double> EzH = std::vector<double>(spacex, 0),
                         EzE = std::vector<double>(spacex, 0),
                         HyE = std::vector<double>(spacex, 0),
                         HyH = std::vector<double>(spacex, 0);
};

struct Field{
    std::vector <double> Hx = std::vector<double>(spacex * spacey * spacez, 0),
                         Hy = std::vector<double>(spacex * spacey * spacez, 0),
                         Hz = std::vector<double>(spacex * spacey * spacez, 0),

                         Ex = std::vector<double>(spacex * spacey * spacez, 0),
                         Ey = std::vector<double>(spacex * spacey * spacez, 0),
                         Ez = std::vector<double>(spacex * spacey * spacez, 0);


    std::vector <double> Hy1d = std::vector<double>(spacex + losslayer, 0),
                         Ez1d = std::vector<double>(spacex + losslayer, 0),
                         Hy1d2 = std::vector<double>(spacex + losslayer, 0),
                         Ez1d2 = std::vector<double>(spacex + losslayer, 0);


    // 6 elements, 3 spacial elements away from border and 2 time elements of
    // those spatial elements
    std::vector <double> Etop = std::vector<double>(3 * 2 * spacex, 0),
                         Ebot = std::vector<double>(3 * 2 * spacex, 0),
                         Eleft = std::vector<double>(3 * 2 * spacey, 0),
                         Eright = std::vector<double>(3 * 2 * spacey, 0),
                         Eforw = std::vector<double>(3 * 2 * spacez, 0),
                         Eback = std::vector<double>(3 * 2 * spacez, 0);

    int t;
};

#define ExH(i, j, k) ExH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define ExE(i, j, k) ExE[(i) + (j) *  spacex + (k) * spacey * spacex]
#define EyH(i, j, k) EyH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define EyE(i, j, k) EyE[(i) + (j) *  spacex + (k) * spacey * spacex]
#define EzH(i, j, k) EzH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define EzE(i, j, k) EzE[(i) + (j) *  spacex + (k) * spacey * spacex]

#define HxH(i, j, k) HxH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define HxE(i, j, k) HxE[(i) + (j) *  spacex + (k) * spacey * spacex]
#define HyH(i, j, k) HyH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define HyE(i, j, k) HyE[(i) + (j) *  spacex + (k) * spacey * spacex]
#define HzH(i, j, k) HzH[(i) + (j) *  spacex + (k) * spacey * spacex]
#define HzE(i, j, k) HzE[(i) + (j) *  spacex + (k) * spacey * spacex]

#define Hx(i, j, k) Hx[(i) + (j) *  spacex + (k) * spacey * spacex]
#define Hy(i, j, k) Hy[(i) + (j) *  spacex + (k) * spacey * spacex]
#define Hz(i, j, k) Hz[(i) + (j) *  spacex + (k) * spacey * spacex]

#define Ex(i, j, k) Ex[(i) + (j) *  spacex + (k) * spacey * spacex]
#define Ey(i, j, k) Ey[(i) + (j) *  spacex + (k) * spacey * spacex]
#define Ez(i, j, k) Ez[(i) + (j) *  spacex + (k) * spacey * spacex]

#define Etop(k, j, i) Etop[(i) * 6 + (j) * 3 + (k)]
#define Ebot(k, j, i) Ebot[(i) * 6 + (j) * 3 + (k)]
#define Eleft(i, j, k) Eleft[(k) * 6 + (j) * 3 + (i)]
#define Eright(i, j, k) Eright[(k) * 6 + (j) * 3 + (i)]
#define Eforw(i, j, k) Eforw[(k) * 6 + (j) * 3 + (i)]
#define Eback(i, j, k) Eback[(k) * 6 + (j) * 3 + (i)]

void FDTD(Field EM,
          const int final_time, const double eps,
          std::ofstream& output);

// Adding ricker solutuion
double ricker(int time, int loc, double Cour);

// Adding plane wave
double planewave(int time, int loc, double Cour, int ppw);

// 2 dimensional functions for E / H movement
void Hupdate3d(Field &EM, Loss &lass, int t);
void Eupdate3d(Field &EM, Loss &lass, int t);

// 1 dimensional update functions for E / H
void Hupdate1d(Field &EM, Loss1d &lass1d, int t);
void Eupdate1d(Field &EM, Loss1d &lass1d, int t);

// Creating loss
void createloss3d(Loss &lass, double eps, double Cour, double loss);
void createloss1d(Loss1d &lass1d, double eps, double Cour, double loss);

// Total Field Scattered Field (TFSF) boundaries
void TFSF(Field &EM, Loss &lass, Loss1d &lass1d, double Cour);
void TFSF2(Field &EM, Loss &lass, Loss1d &lass1d, double Cour);

// Checking Absorbing Boundary Conditions (ABS)
void ABCcheck(Field &EM, Loss &lass);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("3Devanescent.dat", std::ofstream::out);

    int final_time = 501;
    double eps = 377.0;

    // define initial E and H fields
    // std::vector<double> Ez(space, 0.0), Hy(space, 0.0);
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
    double Cour = 1 / sqrt(3);

    Loss lass;
    createloss3d(lass, eps, Cour, loss);
    Loss1d lass1d;
    createloss1d(lass1d, eps, Cour, loss);

    // Time looping
    for (int t = 0; t < final_time; t++){

        Hupdate3d(EM, lass, t);
        //TFSF(EM, lass, lass1d, Cour);
        //TFSF2(EM, lass, lass1d, Cour);
        Eupdate3d(EM,lass,t);
        //ABCcheck(EM, lass);
        EM.Ez(25,25,25) = ricker(t, 0, Cour);

        // Outputting to a file
        int check = 10;
        if (t % check == 0){
            for (size_t dx = 0; dx < spacex; dx++){
                for (size_t dy = 0; dy < spacey; dy++){
                        output << t << '\t' << dx <<'\t' << dy << '\t'
                               << EM.Ez(dx, dy, 25) << '\t' << EM.Hy(dx, dy, 25)
                               << '\t' << EM.Hx(dx, dy, 25) << '\t' << '\n';
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

// 3 dimensional functions for E / H movement
void Hupdate3d(Field &EM, Loss &lass, int t){
    // update magnetic field, x direction
    #pragma omp parallel for
    for (size_t dx = 0; dx < spacex; dx++){
        for (size_t dy = 0; dy < spacey - 1; dy++){
            for (size_t dz = 0; dz < spacez - 1; dz++){
                EM.Hx(dx,dy,dz) = lass.HxH(dx,dy,dz) * EM.Hx(dx,dy,dz)
                                - lass.HxE(dx,dy,dz) * ((EM.Ez(dx,dy+1,dz)
                                                         - EM.Ez(dx,dy,dz))
                                                         - (EM.Ey(dx,dy,dz+1)
                                                         - EM.Ey(dx,dy,dz)));
            }
        }
    }

    // update magnetic field, y direction
    #pragma omp parallel for
    for (size_t dx = 0; dx < spacex - 1; dx++){
        for (size_t dy = 0; dy < spacey; dy++){
            for (size_t dz = 0; dz < spacez - 1; dz++){
                EM.Hy(dx,dy,dz) = lass.HyH(dx,dy,dz) * EM.Hy(dx,dy,dz)
                                - lass.HyE(dx,dy,dz) * ((EM.Ex(dx,dy,dz+1)
                                                      - EM.Ex(dx,dy,dz))
                                                      - (EM.Ez(dx+1,dy,dz)
                                                      - EM.Ez(dx,dy,dz)));
            }
        }
    }

    #pragma omp parallel for
    for (size_t dx = 0; dx < spacex - 1; dx++){
        for (size_t dy = 0; dy < spacey - 1; dy++){
            for (size_t dz = 0; dz < spacez; dz++){
                EM.Hz(dx,dy,dz) = lass.HzH(dx,dy,dz) * EM.Hz(dx,dy,dz)
                                - lass.HzE(dx,dy,dz) * ((EM.Ey(dx+1,dy,dz)
                                                      - EM.Ey(dx,dy,dz))
                                                      - (EM.Ex(dx,dy+1,dz)
                                                      - EM.Ex(dx,dy,dz)));
            }
        }
    }

}


void Eupdate3d(Field &EM, Loss &lass, int t){
    // update electric field
    #pragma omp parallel for
    for (size_t dx = 1; dx < spacex - 1; dx++){
        for (size_t dy = 1; dy < spacey - 1; dy++){
            for (size_t dz = 1; dz < spacez; dz++){
                EM.Ez(dx,dy,dz) = lass.EzE(dx,dy,dz) * EM.Ez(dx,dy,dz)
                                + lass.EzH(dx,dy,dz) * ((EM.Hy(dx,dy,dz)
                                                       - EM.Hy(dx-1,dy,dz))
                                                       - (EM.Hx(dx,dy,dz)
                                                       - EM.Hx(dx,dy-1,dz)));
            }
        }
    }

    #pragma omp parallel for
    for (size_t dx = 1; dx < spacex; dx++){
        for (size_t dy = 1; dy < spacey - 1; dy++){
            for (size_t dz = 1; dz < spacez - 1; dz++){
                EM.Ex(dx,dy,dz) = lass.ExE(dx,dy,dz) * EM.Ex(dx,dy,dz)
                                + lass.ExH(dx,dy,dz) * ((EM.Hz(dx,dy,dz)
                                                       - EM.Hz(dx,dy-1,dz))
                                                       - (EM.Hy(dx,dy,dz)
                                                       - EM.Hy(dx,dy,dz-1)));
            }
        }
    }

    #pragma omp parallel for
    for (size_t dx = 1; dx < spacex - 1; dx++){
        for (size_t dy = 1; dy < spacey; dy++){
            for (size_t dz = 1; dz < spacez - 1; dz++){
                EM.Ey(dx,dy,dz) = lass.EyE(dx,dy,dz) * EM.Ey(dx,dy,dz)
                                + lass.EyH(dx,dy,dz) * ((EM.Hx(dx,dy,dz)
                                                       - EM.Hx(dx,dy,dz-1))
                                                       - (EM.Hz(dx,dy,dz)
                                                       - EM.Hz(dx-1,dy,dz)));
            }
        }
    }

}

// 1 dimensional update functions for E / H
void Hupdate1d(Field &EM, Loss1d &lass1d, int t){
    // update magnetic field, y direction
    #pragma omp parallel for
    for (size_t dx = 0; dx < spacex - 1; dx++){
        EM.Hy1d[dx] = lass1d.HyH[dx] * EM.Hy1d[dx]
                  + lass1d.HyE[dx] * (EM.Ez1d[dx + 1] - EM.Ez1d[dx]);
    }

}

void Eupdate1d(Field &EM, Loss1d &lass1d, int t){
    // update electric field, y direction
    for (size_t dx = 1; dx < spacex - 1; dx++){
        EM.Ez1d[dx] = lass1d.EzE[dx] * EM.Ez1d[dx]
                  + lass1d.EzH[dx] * (EM.Hy1d[dx] - EM.Hy1d[dx - 1]);
    }


}

// Creating loss
void createloss3d(Loss &lass, double eps, double Cour, double loss){

    int buffer = 5;

    for (size_t dx = 0; dx < spacex; dx++){
        for (size_t dy = 0; dy < spacey; dy++){
            for (size_t dz = 0; dz < spacez; dz++){

                // For inhomogeneities add if statements

                    lass.EzH(dx, dy, dz) = Cour * eps;
                    lass.EzE(dx, dy, dz) = 1.0;
                    lass.EyH(dx, dy, dz) = Cour * eps;
                    lass.EyE(dx, dy, dz) = 1.0;
                    lass.ExH(dx, dy, dz) = Cour * eps;
                    lass.ExE(dx, dy, dz) = 1.0;

                    lass.HyE(dx, dy, dz) = Cour * (1.0 / eps);
                    lass.HyH(dx, dy, dz) = 1.0;
                    lass.HxE(dx, dy, dz) = Cour * (1.0 / eps);
                    lass.HxH(dx, dy, dz) = 1.0;
                    lass.HzE(dx, dy, dz) = Cour * (1.0 / eps);
                    lass.HzH(dx, dy, dz) = 1.0;

            }
        }
    }

}

void createloss1d(Loss1d &lass1d, double eps, double Cour, double loss){

    double depth, lossfactor;

    for (size_t dx = 0; dx < spacex; dx++){
        if (dx < spacex - 1 - losslayer){
            lass1d.EzH[dx] = Cour * eps;
            lass1d.EzE[dx] = 1.0;
            lass1d.HyH[dx] = 1.0;
            lass1d.HyE[dx] = Cour / eps;
        }
        else{
            depth = dx - spacex - 1 - losslayer - 0.5;
            lossfactor = loss * pow(depth / (double)losslayer, 2);
            lass1d.EzH[dx] = Cour * eps / (1.0 + lossfactor);
            lass1d.EzE[dx] = (1.0 - lossfactor) / (1.0 + lossfactor);
            depth += 0.5;
            lass1d.HyH[dx] = (1.0 - lossfactor) / (1.0 + lossfactor);
            lass1d.HyE[dx] = Cour / eps / (1.0 + lossfactor);

        }
    }

}


// TFSF boundaries
// We are testing this for a 3d case, not sure if we need to over-update 
// bounds... as in, we might not need to update Hy and Hz in the first loop.
// I am also not sure about the += and -=
void TFSF(Field &EM, Loss &lass, Loss1d &lass1d, double Cour){

    int dx, dy, dz;

    // TFSF boundary
    Bound first, last;
    first.x = 10; last.x = 40;
    first.y = 10; last.y = 40;
    first.z = 10; last.z = 40;

    // Update along right edge!
    dx = last.x;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Hy(dx,dy,dz) += lass.HyE(dx, dy, dz) * EM.Ez1d[dx];
            EM.Hz(dx,dy,dz) += lass.HzE(dx, dy, dz) * EM.Ez1d[dx];
        }
    }

    // Updating along left edge
    dx = first.x - 1;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Hy(dx,dy,dz) -= lass.HyE(dx, dy, dz) * EM.Ez1d[dx+1];
            EM.Hz(dx,dy,dz) -= lass.HzE(dx, dy, dz) * EM.Ez1d[dx+1];

        }
    }

    // Updating along top
    dy = last.y;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Hx(dx,dy,dz) -= lass.HxE(dx, dy, dz) * EM.Ez1d[dx];
            EM.Hz(dx,dy,dz) -= lass.HzE(dx, dy, dz) * EM.Ez1d[dx];
        }
    }

    // Update along bot
    dy = first.y - 1;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Hx(dx,dy,dz) += lass.HxE(dx, dy, dz) * EM.Ez1d[dx];
            EM.Hz(dx,dy,dz) += lass.HzE(dx, dy, dz) * EM.Ez1d[dx];

        }
    }

    // Update along forw
    dz = first.z - 1;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Hx(dx,dy,dz) -= lass.HxE(dx, dy, dz) * EM.Ez1d[dx];
            EM.Hy(dx,dy,dz) -= lass.HyE(dx, dy, dz) * EM.Ez1d[dx];
        }
    }

    // Update along back
    dz = last.z;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dy = first.y; dy <= last.y; dy++){
            EM.Hx(dx,dy,dz) += lass.HxE(dx, dy, dz) * EM.Ez1d[dx];
            EM.Hy(dx,dy,dz) += lass.HyE(dx, dy, dz) * EM.Ez1d[dx];
        }
    }



    // Insert 1d grid stuff here. Update magnetic and electric field
    Hupdate1d(EM, lass1d, EM.t);
    Eupdate1d(EM, lass1d, EM.t);
    //EM.Ez1d[10] = ricker(EM.t,0, Cour);
    //EM.Ez1d[10] = planewave(EM.t, 15, Cour, 10);
    //EM.Ez1d[290] = planewave(EM.t, 15, Cour, 10);
    EM.t++;

    // Check mag instead of ricker.
    // Update along right
    dx = last.x;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Ez(dx,dy,dz) += lass.EzH(dx,dy,dz) * EM.Hy1d[dx];
            EM.Ey(dx,dy,dz) += lass.EyH(dx,dy,dz) * EM.Hy1d[dx];
        }
    }

    // Updating Ez along left
    dx = first.x;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Ez(dx,dy,dz) -= lass.EzH(dx,dy,dz) * EM.Hy1d[dx-1];
            EM.Ey(dx,dy,dz) -= lass.EyH(dx,dy,dz) * EM.Hy1d[dx-1];
        }
    }

    dy = last.y;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Ex(dx,dy,dz) += lass.ExH(dx,dy,dz) * EM.Hy1d[dx];
            EM.Ez(dx,dy,dz) += lass.EzH(dx,dy,dz) * EM.Hy1d[dx];
        }
    }

    // Updating Ez along left
    dy = first.y;
    #pragma omp parallel for
    for (int dx = first.x; dx <= last.x; dx++){
        for (int dz = first.z; dz <= last.z; dz++){
            EM.Ez(dx,dy,dz) -= lass.EzH(dx,dy,dz) * EM.Hy1d[dx];
            EM.Ex(dx,dy,dz) -= lass.ExH(dx,dy,dz) * EM.Hy1d[dx];
        }
    }

    // Updating along back
    dz = last.z;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dx = first.x; dx <= last.x; dx++){
            EM.Ex(dx,dy,dz) += lass.ExH(dx,dy,dz) * EM.Hy1d[dx];
            EM.Ey(dx,dy,dz) += lass.EyH(dx,dy,dz) * EM.Hy1d[dx];
        }
    }

    // Updating Ez along forw
    dz = first.z;
    #pragma omp parallel for
    for (int dy = first.y; dy <= last.y; dy++){
        for (int dx = first.x; dx <= last.x; dx++){
            EM.Ex(dx,dy,dz) -= lass.ExH(dx,dy,dz) * EM.Hy1d[dx];
            EM.Ey(dx,dy,dz) -= lass.EyH(dx,dy,dz) * EM.Hy1d[dx];
        }
    }


}

// Checking Absorbing Boundary Conditions (ABC)
// Adding multiple fileds different polarization possibilities.
// note: running int he TMz polarization, so we will memorize Ez at end.
void ABCcheck(Field &EM, Loss &lass){

    // defining constant for  ABC
    double c1, c2, c3, temp1, temp2;
    temp1 = sqrt(lass.EzH(0,0,0) * lass.HyE(0,0,0));
    temp2 = 1.0 / temp1 + 2.0 + temp1;
    c1 = -(1.0 / temp1 - 2.0 + temp1) / temp2;
    c2 = -2.0 * (temp1 - 1.0 / temp1) / temp2;
    c3 = 4.0 * (temp1 + 1.0 / temp1) / temp2;
    size_t dx, dy, dz;

    // Setting ABC for top
    #pragma omp parallel for
    for (dx = 0; dx < spacex; dx++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ez(dx,spacey-1,dz) = c1 * (EM.Ez(dx, spacey - 3,dz) 
                          + EM.Etop(0, 1, dx))
                          + c2 * (EM.Etop(0, 0, dx) + EM.Etop(2, 0 , dx)
                                  -EM.Ez(dx,spacey - 2,dz) -EM.Etop(1, 1, dx))
                          + c3 * EM.Etop(1, 0, dx) - EM.Etop(2, 1, dx);

            EM.Ex(dx,spacey-1,dz) = c1 * (EM.Ex(dx, spacey - 3,dz) 
                          + EM.Etop(0, 1, dx))
                          + c2 * (EM.Etop(0, 0, dx) + EM.Etop(2, 0 , dx)
                                  -EM.Ex(dx,spacey - 2,dz) -EM.Etop(1, 1, dx))
                          + c3 * EM.Etop(1, 0, dx) - EM.Etop(2, 1, dx);

            // memorizing fields...
            for (dy = 0; dy < 3; dy++){
                EM.Etop(dy, 1, dx) = EM.Etop(dy, 0, dx);
                EM.Etop(dy, 0, dx) = EM.Ez(dx, spacey - 1 - dy,dx);
            }
        }
    }

    // Setting ABC for bottom
    #pragma omp parallel for
    for (dx = 0; dx < spacex; dx++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ez(dx,0,dz) = c1 * (EM.Ez(dx, 2,dz) + EM.Ebot(0, 1, dx))
                          + c2 * (EM.Ebot(0, 0, dx) + EM.Ebot(2, 0 , dx)
                                  -EM.Ez(dx,1,dz) -EM.Ebot(1, 1, dx))
                          + c3 * EM.Ebot(1, 0, dx) - EM.Ebot(2, 1, dx);

            EM.Ex(dx,0,dz) = c1 * (EM.Ex(dx, 2,dz) + EM.Ebot(0, 1, dx))
                          + c2 * (EM.Ebot(0, 0, dx) + EM.Ebot(2, 0 , dx)
                                  -EM.Ex(dx,1,dz) -EM.Ebot(1, 1, dx))
                          + c3 * EM.Ebot(1, 0, dx) - EM.Ebot(2, 1, dx);


            // memorizing fields...
            for (dy = 0; dy < 3; dy++){
                EM.Ebot(dy, 1, dx) = EM.Ebot(dy, 0, dx);
                EM.Ebot(dy, 0, dx) = EM.Ez(dx, dy, dz);
            }
        }
    }

    // ABC on right
    #pragma omp parallel for
    for (dy = 0; dy < spacey; dy++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ez(spacex - 1,dy,dz) = c1 * (EM.Ez(spacex - 3,dy,dz) 
                                  + EM.Eright(0, 1, dy))
                          + c2 * (EM.Eright(0, 0, dy) + EM.Eright(2, 0 , dy)
                                  -EM.Ez(spacex - 2,dy,dz) -EM.Eright(1, 1, dy))
                          + c3 * EM.Eright(1, 0, dy) - EM.Eright(2, 1, dy);

            EM.Ey(spacex - 1,dy,dz) = c1 * (EM.Ey(spacex - 3,dy,dz) 
                                  + EM.Eright(0, 1, dy))
                          + c2 * (EM.Eright(0, 0, dy) + EM.Eright(2, 0 , dy)
                                  -EM.Ey(spacex - 2,dy,dz) -EM.Eright(1, 1, dy))
                          + c3 * EM.Eright(1, 0, dy) - EM.Eright(2, 1, dy);

            // memorizing fields...
            for (dx = 0; dx < 3; dx++){
                EM.Eright(dx, 1, dy) = EM.Eright(dx, 0, dy);
                EM.Eright(dx, 0, dy) = EM.Ez(spacex - 1 - dx, dy, dz);
            }
        }
    }

    // Setting ABC for left side of grid. Woo!
    #pragma omp parallel for
    for (dy = 0; dy < spacey; dy++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ez(0,dy,dz) = c1 * (EM.Ez(2,dy,dz) + EM.Eleft(0, 1, dy))
                          + c2 * (EM.Eleft(0, 0, dy) + EM.Eleft(2, 0 , dy)
                                  -EM.Ez(1,dy,dz) -EM.Eleft(1, 1, dy))
                          + c3 * EM.Eleft(1, 0, dy) - EM.Eleft(2, 1, dy);

            EM.Ey(0,dy,dz) = c1 * (EM.Ey(2,dy,dz) + EM.Eleft(0, 1, dy))
                          + c2 * (EM.Eleft(0, 0, dy) + EM.Eleft(2, 0 , dy)
                                  -EM.Ey(1,dy,dz) -EM.Eleft(1, 1, dy))
                          + c3 * EM.Eleft(1, 0, dy) - EM.Eleft(2, 1, dy);


            // memorizing fields...
            for (dx = 0; dx < 3; dx++){
                EM.Eleft(dx, 1, dy) = EM.Eleft(dx, 0, dy);
                EM.Eleft(dx, 0, dy) = EM.Ez(dx, dy, dz);
            }
        }
    }

    // ABC on back
    #pragma omp parallel for
    for (dy = 0; dy < spacey; dy++){
        for (dx = 0; dx < spacex; dx++){
            EM.Ex(dx,dy,spacez-1) = c1 * (EM.Ex(spacex - 3,dy,dz) 
                                  + EM.Eback(0, 1, dy))
                          + c2 * (EM.Eback(0, 0, dy) + EM.Eback(2, 0 , dy)
                                  -EM.Ex(spacex - 2,dy,dz) -EM.Eback(1, 1, dy))
                          + c3 * EM.Eback(1, 0, dy) - EM.Eback(2, 1, dy);

            EM.Ey(dy,dy,spacez-1) = c1 * (EM.Ey(spacex - 3,dy,dz) 
                                  + EM.Eback(0, 1, dy))
                          + c2 * (EM.Eback(0, 0, dy) + EM.Eback(2, 0 , dy)
                                  -EM.Ey(spacex - 2,dy,dz) -EM.Eback(1, 1, dy))
                          + c3 * EM.Eback(1, 0, dy) - EM.Eback(2, 1, dy);


            // memorizing fields...
            for (dz = 0; dz < 3; dz++){
                EM.Eback(dy, 1, dz) = EM.Eback(dx, 0, dy);
                EM.Eback(dy, 0, dz) = EM.Ez(spacex - 1 - dx, dy, dz);
            }
        }
    }

    // Setting ABC for forw
    #pragma omp parallel for
    for (dy = 0; dy < spacey; dy++){
        for (dx = 0; dx < spacex; dx++){
            EM.Ex(dx,dy,0) = c1 * (EM.Ex(2,dy,dz) + EM.Etop(0, 1, dy))
                          + c2 * (EM.Etop(0, 0, dy) + EM.Etop(2, 0 , dy)
                                  -EM.Ex(1,dy,dz) -EM.Etop(1, 1, dy))
                          + c3 * EM.Etop(1, 0, dy) - EM.Etop(2, 1, dy);

            EM.Ey(dx,dy,0) = c1 * (EM.Ey(2,dy,dz) + EM.Etop(0, 1, dy))
                          + c2 * (EM.Etop(0, 0, dy) + EM.Etop(2, 0 , dy)
                                  -EM.Ey(1,dy,dz) -EM.Etop(1, 1, dy))
                          + c3 * EM.Etop(1, 0, dy) - EM.Etop(2, 1, dy);


            // memorizing fields...
            for (dz = 0; dz < 3; dz++){
                EM.Etop(dy, 1, dz) = EM.Etop(dx, 0, dy);
                EM.Etop(dy, 0, dz) = EM.Ez(dx, dy, dz);
            }
        }
    }

}


// Adding plane wave
double planewave(int time, int loc, double Cour, int ppw){
    double plane;

    plane = sin((1 / (double)ppw) * (Cour * (double)time -
                 (double)loc));
    //plane = sin((double)(time-loc) * 3.5 / radius);

    return plane;
}

