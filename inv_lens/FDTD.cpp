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

struct Loss1d{
    std::vector <double> EzH = std::vector<double>(space, 0), 
                         EzE = std::vector<double>(space, 0), 
                         HyE = std::vector<double>(space, 0), 
                         HyH = std::vector<double>(space, 0),
                         HxE = std::vector<double>(space, 0), 
                         HxH = std::vector<double>(space, 0);
};

struct Field{
    std::vector <double> Hx = std::vector<double>(space * space, 0), 
                         Hy = std::vector<double>(space * space, 0),
                         Ez = std::vector<double>(space * space, 0);
    int t;
};

struct Field1d{
    std::vector <double> Hx = std::vector<double>(space, 0), 
                         Hy = std::vector<double>(space, 0),
                         Ez = std::vector<double>(space, 0);
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

void FDTD(Field EM, Field1d EM1d,
          const int final_time, const double eps, const int space,
          std::ofstream& output);

// Adding ricker solutuion
double ricker(int time, int loc, double Cour);

// 2 dimensional functions for E / H movement
Field Hupdate2d(Field EM, Loss lass, int t);
Field Eupdate2d(Field EM, Loss lass, int t);

// 1 dimensional update functions for E / H
Field1d Hupdate1d(Field1d EM1d, Loss lass, int t);
Field1d Eupdate1d(Field1d EM1d, Loss lass, int t);

// Creating loss
Loss createloss2d(Loss lass, double eps, double Cour, double loss);
Loss1d createloss1d(Loss1d lass1d, double eps, double Cour, double loss);

// TFSF boundaries
Field TFSF(Field EM, Loss lass, Field1d EM1d, Loss1d lass1d, double Cour);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("FDTD.dat", std::ofstream::out);

    int final_time = 1000;
    double eps = 377.0;

    // define initial E and H fields
    // std::vector<double> Ez(space, 0.0), Hy(space, 0.0);
    Field EM;
    Field1d EM1d;
    EM.t = 0;

    FDTD(EM, EM1d, final_time, eps, space, output);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// This is the function we writs the bulk of the code in
void FDTD(Field EM, Field1d EM1d,
          const int final_time, const double eps, const int space,
          std::ofstream& output){

    // For magnetic field:
    // double offset = 0.00005;

    // for electric field:
    //double offset = 0.05;
    double loss = 0.0;
    double Cour = 1.0 / sqrt(2.0);

    Loss lass;
    lass  = createloss2d(lass, eps, Cour, loss);
    Loss1d lass1d;
    lass1d = createloss1d(lass1d, eps, Cour, loss);

    // Time looping
    for (int t = 0; t < final_time; t++){

        EM = Hupdate2d(EM, lass, t);
        EM = TFSF(EM, lass, EM1d, lass1d, Cour);
        EM = Eupdate2d(EM,lass,t);
        //EM.Ez(0,100) = ricker(t, 0, Cour);
        
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

// 2 dimensional functions for E / H movement
Field Hupdate2d(Field EM, Loss lass, int t){
    // update magnetic field, x direction
    for (size_t dx = 0; dx < space; dx++){
        for (size_t dy = 0; dy < space - 1; dy++){
           EM.Hx(dx,dy) = lass.HxH(dx,dy) * EM.Hx(dx, dy) 
                       - lass.HxE(dx,dy) * (EM.Ez(dx,dy + 1) 
                                            - EM.Ez(dx,dy));
        }
    }


    // update magnetic field, y direction
    for (size_t dx = 0; dx < space - 1; dx++){
        for (size_t dy = 0; dy < space; dy++){
           EM.Hy(dx,dy) = lass.HyH(dx,dy) * EM.Hy(dx,dy) 
                      + lass.HyE(dx,dy) * (EM.Ez(dx + 1,dy) 
                                            - EM.Ez(dx,dy));
        }
    }

    return EM;

}


Field Eupdate2d(Field EM, Loss lass, int t){
    // update electric field
    for (size_t dx = 1; dx < space - 1; dx++){
        for (size_t dy = 1; dy < space - 1; dy++){
           EM.Ez(dx,dy) = lass.EzE(dx,dy) * EM.Ez(dx,dy)
                       + lass.EzH(dx,dy) * ((EM.Hy(dx, dy)
                                         - EM.Hy(dx - 1, dy))
                                         - (EM.Hx(dx,dy)
                                         - EM.Hx(dx, dy - 1)));
        }
    }
    return EM;
}

// 1 dimensional update functions for E / H
Field1d Hupdate1d(Field1d EM1d, Loss1d lass1d, int t){
    // update magnetic field, y direction
    for (size_t dx = 0; dx < space - 1; dx++){
        EM1d.Hy[dx] = lass1d.HyH[dx] * EM1d.Hy[dx] 
                  + lass1d.HyE[dx] * (EM1d.Ez[dx + 1] - EM1d.Ez[dx]);
    }

    return EM1d;
}

Field1d Eupdate1d(Field1d EM1d, Loss1d lass1d, int t){
    // update electric field, y direction
    for (size_t dx = 1; dx < space - 1; dx++){
        EM1d.Ez[dx] = lass1d.EzE[dx] * EM1d.Ez[dx] 
                  + lass1d.EzH[dx] * (EM1d.Hy[dx] - EM1d.Hy[dx - 1]);
    }

    return EM1d;

}

// Creating loss
Loss createloss2d(Loss lass, double eps, double Cour, double loss){
    for (size_t dx = 0; dx < space; dx++){
        for (size_t dy = 0; dy < space; dy++){
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


    return lass;
}
Loss1d createloss1d(Loss1d lass1d, double eps, double Cour, double loss){
    for (size_t dx = 0; dx < space; dx++){
            //if (dx > 100 && dx < 150){

            lass1d.EzH[dx ]= Cour * eps;
            lass1d.EzE[dx] = 1.0;
            lass1d.HyH[dx] = 1.0;
            lass1d.HyE[dx] = Cour / eps;
/*

            lass1d.EzH[dx] =  eps / 9.0 /(1.0 - loss);
            lass1d.EzE[dx] = (1.0 - loss) / (1.0 + loss);
            lass1d.HyH[dx] = (1.0 - loss) / (1.0 + loss);
*/
            lass1d.HyE[dx] = (1.0 / eps) / (1.0 + loss);
        //}
        /*
        else{
            lass1d.EzH[dx] =  Cour * eps;
            lass1d.EzE[dx] = 1.0;
            lass1d.HyH[dx] = 1.0;
            lass1d.HyE[dx] = Cour / eps;
                
            lass1d.EzH[dx] =  eps;
            lass1d.EzE[dx] = 1.0;
            lass1d.HyH[dx] = 1.0;
            lass1d.HyE[dx] = (1.0 / eps);
                
        }
        */
    }


    return lass1d;

}

// TFSF boundaries
Field TFSF(Field EM, Loss lass, Field1d EM1d, Loss1d lass1d, double Cour){
    // TFSF boundary
    Bound first, last;
    first.x = 50; last.x = 150;
    first.y = 50; last.y = 150;

    // Updating along left edge
    int dx = first.x - 1;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Hy(dx,dy) -= lass.HyE(dx, dy) * EM1d.Ez[dx+1];
    }

    // Update along right edge!
    dx = last.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Hy(dx,dy) += lass.HyE(dx, dy) * EM1d.Ez[dx];
    }

    // Update along bot
    int dy = first.y - 1;
    for (int dx = first.x; dx <= last.x; dx++){
        EM.Hx(dx,dy) += lass.HxE(dx, dy) * EM1d.Ez[dx];
    }

    // Updating along top
    dy = last.y;
    for (int dx = first.x; dx <= last.x; dx++){
        EM.Hx(dx,dy) -= lass.HxE(dx, dy) * EM1d.Ez[dx];
    }

    // Insert 1d grid stuff here. Update magnetic and electric field
    Hupdate1d(EM1d, lass1d, EM.t);
    Eupdate1d(EM1d, lass1d, EM.t);
    EM1d.Ez[100] = ricker(EM.t,0, Cour);
    EM.t++;
    std::cout << EM.t << '\n';

    // Check mag instead of ricker.
    // Updating Ez along left
    dx = first.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Ez(dx, dy) -= lass.EzH(dx, dy) * EM1d.Hy[dx - 1];
    }

    // Update along right
    dx = last.x;
    for (int dy = first.y; dy <= last.y; dy++){
        EM.Ez(dx, dy) += lass.EzH(dx, dy) * EM1d.Hy[dx];
    }

    return EM;

}
