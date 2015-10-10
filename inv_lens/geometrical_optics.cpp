/*-------------geometrical_optics.cpp-----------------------------------------//
*
*              geometrical optics
*
* Purpose: to simulate light going through a lens with a variable refractive 
*          index. Not wave-like.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <cmath>
#include <fstream>
#include <cassert>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// constants
const int lightnum = 10, time_res = 200;

// Struct for space
struct dimensions{
    double x, y;
};

// Struct for light rays
struct light_rays{
   dimensions ray[lightnum];
   dimensions ray_vel[lightnum];
   double index[lightnum];
};

// checks refractive index profile
double check_n(double x, double y);

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double angle);

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel,
                     std::ofstream &output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("out.dat", std::ofstream::out);

    dimensions dim;
    dim.x = 4;
    dim.y = 10;

    double max_vel = 1;
    light_rays pvec = light_gen(dim, max_vel, 0.523598776);

    pvec = propagate(pvec, 0.1, max_vel, output);

    output << "\n \n5	0	0	0 \n5	5	0	0 \n";

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// checks refractive index profile
double check_n(double x, double y){
    // check refractive index against a set profile

    double index;

    if (x < 5 || x > 7){
        index = 1.0;
    }
    else{
        index = (x - 3.0);
        //index = 1.4;
    }

    return index;
}

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double angle){

    light_rays ray_diagram;

    // create rays
    for (size_t i = 0; i < lightnum; i++){
        ray_diagram.ray[i].x = (double)i * dim.x / lightnum;
        ray_diagram.ray[i].y = 0;
        //ray_diagram.ray[i].y = cos(angle);
        //ray_diagram.ray[i].x = sin(angle);
        ray_diagram.ray_vel[i].x = max_vel * cos(angle);
        ray_diagram.ray_vel[i].y = max_vel * sin(angle);
        ray_diagram.index[i] = check_n(ray_diagram.ray[i].x, 
                                       ray_diagram.ray[i].y);
    }

    return ray_diagram;
}

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel,
                     std::ofstream &output){

    double index_p, theta, theta_p;
    double iratio, dotprod;

    // move simulation every timestep
    for (size_t i = 0; i < time_res; i++){
        for (size_t j = 0; j < lightnum; j++){
            ray_diagram.ray[j].x += ray_diagram.ray_vel[j].x * step_size; 
            ray_diagram.ray[j].y += ray_diagram.ray_vel[j].y * step_size;
            if (ray_diagram.index[j] != 
                check_n(ray_diagram.ray[j].x, ray_diagram.ray[j].y)){
                index_p = check_n(ray_diagram.ray[j].x,
                                  ray_diagram.ray[j].y);

                std::cout << index_p << '\t' << i << '\t' << j << '\n';
                //assert(isnan(ray_diagram.index[j]));

                theta = atan2(ray_diagram.ray_vel[j].y, 
                              ray_diagram.ray_vel[j].x);
                theta_p = asin((ray_diagram.index[j] / index_p) * sin(theta));
                ray_diagram.ray_vel[j].y = max_vel * sin(theta_p);
                ray_diagram.ray_vel[j].x = max_vel * cos(theta_p);
/*
                double eta, NdotI;

                // Index of refraction
                eta = ray_diagram.index[j] / index_p;
				
                // Ray direction vector, normalized
                double mag = std::sqrt(ray_diagram.ray_vel[j].x * 
                                       ray_diagram.ray_vel[j].x +
                                       ray_diagram.ray_vel[j].y * 
                                       ray_diagram.ray_vel[j].y);

                double Ix = ray_diagram.ray_vel[j].x;
                double Iy = ray_diagram.ray_vel[j].y;

                // The index of refraction normal vector is [1,0].
                NdotI = -Ix;

                // Choose reflection or refraction, in a physically correct 
                //model there would be all sorts of
                // probability calculations and possibility to both reflect 
                // and refract at once.
                float k = 1.0 - eta * eta * (1.0 - NdotI * NdotI);
                    if (k < 0.0)
                    {
                    // total internal reflection, reflect here
                    }
                    else
                    {
                        // refraction
                        // Normal = -1 in x dir
                        ray_diagram.ray_vel[j].x = eta * Ix + 
                                                  (eta * NdotI - std::sqrt(k)) 
                                                   * -1.0; 
                        // Normal = 0 in y dir
                        ray_diagram.ray_vel[j].y = eta * Iy + 
                                                  (eta * NdotI - std::sqrt(k)) 
                                                   * 0.0;
                    }
*/
/*
                double r = ray_diagram.index[j] / index_p;
                double mag = std::sqrt(ray_diagram.ray_vel[j].x * 
                                       ray_diagram.ray_vel[j].x +
                                       ray_diagram.ray_vel[j].y * 
                                       ray_diagram.ray_vel[j].y); 


                double ix = ray_diagram.ray_vel[j].x / mag;
                double iy = ray_diagram.ray_vel[j].y / mag;

                // Normal: [-1, 0]
                double c = -ix;

                double k = 1.0 - r * r * (1.0 - c * c);

                if (k < 0.0) {
                    // Do whatever
                } else {
                    double k1 = std::sqrt(k);
                    ray_diagram.ray_vel[j].x = r * ix - (r * c - k1);
                    ray_diagram.ray_vel[j].y = r * iy;
                }
*/
                ray_diagram.index[j] = index_p;
            }

        }

        for (size_t q = 0; q < lightnum; q++){
            output << ray_diagram.ray[q].x <<'\t'<< ray_diagram.ray[q].y << '\t'
                   << ray_diagram.ray_vel[q].x <<'\t'<< ray_diagram.ray_vel[q].y
                   << '\n';
        }

        output << '\n' << '\n';
    }

    return ray_diagram;
}

