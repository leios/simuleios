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

struct Bound_pos{
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
    std::vector <double> Eyx0 = std::vector<double>(spacez * spacey, 0),
                         Ezx0 = std::vector<double>(spacez * spacey, 0),
                         Eyx1 = std::vector<double>(spacez * spacey, 0),
                         Ezx1 = std::vector<double>(spacez * spacey, 0),

                         Exy0 = std::vector<double>(spacez * spacex, 0),
                         Ezy0 = std::vector<double>(spacez * spacex, 0),
                         Exy1 = std::vector<double>(spacez * spacex, 0),
                         Ezy1 = std::vector<double>(spacez * spacex, 0),

                         Exz0 = std::vector<double>(spacex * spacey, 0),
                         Eyz0 = std::vector<double>(spacex * spacey, 0),
                         Exz1 = std::vector<double>(spacez * spacey, 0),
                         Eyz1 = std::vector<double>(spacez * spacey, 0);

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

#define Eyx0(k, j) Eyx0[(k) * spacey + (j)]
#define Ezx0(k, j) Ezx0[(k) * (spacez - 1) + (j)]
#define Eyx1(k, j) Eyx1[(k) * spacey + (j)]
#define Ezx1(k, j) Ezx1[(k) * (spacez - 1) + (j)]

#define Exy0(k, j) Exy0[(k) * spacex + (j)]
#define Ezy0(k, j) Ezy0[(k) * (spacez - 1) + (j)]
#define Exy1(k, j) Exy1[(k) * spacey + (j)]
#define Ezy1(k, j) Ezy1[(k) * (spacez - 1) + (j)]

#define Exz0(k, j) Exz0[(k) * spacex + (j)]
#define Eyz0(k, j) Eyz0[(k) * (spacey - 1) + (j)]
#define Exz1(k, j) Exz1[(k) * spacex + (j)]
#define Eyz1(k, j) Eyz1[(k) * (spacey - 1) + (j)]

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

// Checking Absorbing Boundary Conditions (ABS)
void ABCcheck(Field &EM, Loss &lass, double Cour);

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
        Eupdate3d(EM,lass,t);
        ABCcheck(EM, lass, Cour);
        EM.Ez(25,25,25) = 100 * ricker(t, 0, Cour);

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
    Bound_pos first, last;
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
    EM.Ez1d[10] = ricker(EM.t,0, Cour);
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
// note: running in the TMz polarization, so we will memorize Ez at end.
//       Also: combine loops, if possible!
void ABCcheck(Field &EM, Loss &lass, double Cour){

    double abccoef = (Cour - 1.0) / (Cour + 1.0);
    size_t dx, dy, dz;

    // ABC at x0
    dx = 0;
    for (dy = 0; dy < spacey - 1; dy++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ey(dx,dy,dz) = EM.Eyx0(dy,dz) 
                              + abccoef*(EM.Ey(dx+1,dy,dz)-EM.Ey(dx,dy,dz));
            EM.Eyx0(dy,dz) = EM.Ey(dx+1,dy,dz);
        }
    }
    for (dy = 0; dy < spacey; dy++){
        for (dz = 0; dz < spacez - 1; dz++){
            EM.Ez(dx,dy,dz) = EM.Ezx0(dy,dz) 
                              + abccoef*(EM.Ez(dx+1,dy,dz)-EM.Ez(dx,dy,dz));
            EM.Ezx0(dy,dz) = EM.Ez(dx+1,dy,dz);
        }
    }

    // ABC at x1
    dx = spacex - 1;
    for (dy = 0; dy < spacey - 1; dy++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ey(dx,dy,dz) = EM.Eyx1(dy,dz) 
                              + abccoef*(EM.Ey(dx-1,dy,dz)-EM.Ey(dx,dy,dz));
            EM.Eyx1(dy,dz) = EM.Ey(dx-1,dy,dz);
        }
    }
    for (dy = 0; dy < spacey; dy++){
        for (dz = 0; dz < spacez - 1; dz++){
            EM.Ez(dx,dy,dz) = EM.Ezx1(dy,dz) 
                              + abccoef*(EM.Ez(dx-1,dy,dz)-EM.Ez(dx,dy,dz));
            EM.Ezx1(dy,dz) = EM.Ez(dx-1,dy,dz);
        }
    }

    // ABC at y0
    dy = 0;
    for (dx = 0; dx < spacex - 1; dx++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ex(dx,dy,dz) = EM.Exy0(dx,dz) 
                              + abccoef*(EM.Ex(dx,dy+1,dz)-EM.Ex(dx,dy,dz));
            EM.Exy0(dx,dz) = EM.Ex(dx,dy+1,dz);
        }
    }
    for (dx = 0; dx < spacey; dx++){
        for (dz = 0; dz < spacez - 1; dz++){
            EM.Ez(dx,dy,dz) = EM.Ezy0(dx,dz) 
                              + abccoef*(EM.Ez(dx,dy+1,dz)-EM.Ez(dx,dy,dz));
            EM.Ezy0(dx,dz) = EM.Ez(dx,dy+1,dz);
        }
    }

    // ABC at y1
    dy = spacey - 1;
    for (dx = 0; dx < spacex - 1; dx++){
        for (dz = 0; dz < spacez; dz++){
            EM.Ex(dx,dy,dz) = EM.Exy1(dx,dz) 
                              + abccoef*(EM.Ex(dx,dy-1,dz)-EM.Ex(dx,dy,dz));
            EM.Exy1(dx,dz) = EM.Ex(dx,dy-1,dz);
        }
    }
    for (dx = 0; dx < spacey; dx++){
        for (dz = 0; dz < spacez - 1; dz++){
            EM.Ez(dx,dy,dz) = EM.Ezy1(dx,dz) 
                              + abccoef*(EM.Ez(dx,dy-1,dz)-EM.Ez(dx,dy,dz));
            EM.Ezy1(dx,dz) = EM.Ez(dx,dy-1,dz);
        }
    }

    // ABC at z0
    dz = 0;
    for (dx = 0; dx < spacex - 1; dx++){
        for (dy = 0; dy < spacey; dy++){
            EM.Ex(dx,dy,dz) = EM.Exz0(dx,dy) 
                              + abccoef*(EM.Ex(dx,dy,dz+1)-EM.Ex(dx,dy,dz));
            EM.Exz0(dx,dy) = EM.Ex(dx,dy,dz+1);
        }
    }
    for (dx = 0; dx < spacey; dx++){
        for (dy = 0; dy < spacey - 1; dy++){
            EM.Ey(dx,dy,dz) = EM.Eyz0(dx,dy) 
                              + abccoef*(EM.Ey(dx,dy,dz+1)-EM.Ey(dx,dy,dz));
            EM.Eyz0(dx,dy) = EM.Ey(dx,dy,dz+1);
        }
    }

    // ABC at z1
    dz = spacez - 1;
    for (dx = 0; dx < spacex - 1; dx++){
        for (dy = 0; dy < spacey; dy++){
            EM.Ex(dx,dy,dz) = EM.Exz1(dx,dy) 
                              + abccoef*(EM.Ex(dx,dy,dz-1)-EM.Ex(dx,dy,dz));
            EM.Exz1(dx,dy) = EM.Ex(dx,dy,dz-1);
        }
    }
    for (dx = 0; dx < spacey; dx++){
        for (dy = 0; dy < spacey - 1; dy++){
            EM.Ey(dx,dy,dz) = EM.Eyz1(dx,dy) 
                              + abccoef*(EM.Ey(dx,dy,dz-1)-EM.Ey(dx,dy,dz));
            EM.Eyz1(dx,dy) = EM.Ey(dx,dy,dz-1);
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

