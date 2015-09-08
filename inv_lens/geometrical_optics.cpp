/*-------------geometrical_optics.cpp-----------------------------------------//
*
*              geometrical optics
*
* Purpose: to simulate light going through a lens with a variable refractive 
*          index. Not wave-like.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <math.h>
#include <fstream>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// constants
const int lightnum = 10, time_res = 100;

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
    dim.x = 10;
    dim.y = 10;

    double max_vel = 1;
    light_rays pvec = light_gen(dim, max_vel, 0.523598776);

    pvec = propagate(pvec, 0.1, max_vel, output);


}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// checks refractive index profile
double check_n(double x, double y){
    // check refractive index against a set profile

    double index;

    if (x < 5){
        index = 1.0;
    }
    else{
        index = 1.4;
    }

    return index;
}

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double angle){

    light_rays ray_diagram;

    // create rays
    for (size_t i = 0; i < lightnum; i++){
        //ray_diagram.ray[i].x = (double)i * dim.y / lightnum;
        //ray_diagram.ray[i].y = 0;
        ray_diagram.ray[i].x = cos(angle);
        ray_diagram.ray[i].y = sin(angle);
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
/*
                index_p = check_n(ray_diagram.ray[j].x,
                                  ray_diagram.ray[j].y);

                theta = atan2(ray_diagram.ray_vel[j].y, 
                              ray_diagram.ray_vel[j].x);
                theta_p = asin((ray_diagram.index[j] / index_p) * sin(theta));
                ray_diagram.ray_vel[j].x = max_vel * sin(theta_p);
                ray_diagram.ray_vel[j].y = max_vel * cos(theta_p);
*/
                iratio = ray_diagram.index[j] / index_p;

                // The index of refraction normal vector is [1,0].
                // Note that this must be updated for future videos!
                dotprod = -(ray_diagram.ray_vel[j].x);

                ray_diagram.ray_vel[j].x = iratio * ray_diagram.ray_vel[j].x +
                                          (iratio * dotprod -
                                           sqrt(1 - iratio * iratio * 
                                           (1 - dotprod * dotprod)));
                ray_diagram.ray_vel[j].y = iratio * ray_diagram.ray_vel[j].y;

                ray_diagram.index[j] = index_p;
                
            }

        }
        output << ray_diagram.ray[0].x << '\t' << ray_diagram.ray[0].y << '\n';
    }

    return ray_diagram;
}

