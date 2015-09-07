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
light_rays light_gen(dimensions dim, double max_vel, double ratio);

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// checks refractive index profile
double check_n(double x, double y){
    // check refractive index against a set profile

    double index;

    if (x < 10){
        index = 1.0;
    }
    else{
        index = 1.4;
    }

    return index;
}

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double ratio){

    light_rays ray_diagram;

    // create rays
    for (size_t i = 0; i < lightnum; i++){
        ray_diagram.ray[i].x = (double)i * dim.y / lightnum;
        ray_diagram.ray[i].y = 0;
        ray_diagram.ray_vel[i].x = max_vel;
        ray_diagram.ray_vel[i].y = 0;
        ray_diagram.index[i] = check_n(ray_diagram.ray[i].x, 
                                       ray_diagram.ray[i].y);
    }

    return ray_diagram;
}

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel){

    double index_p, theta, theta_p;

    // move simulation every timestep
    for (size_t i = 0; i < time_res; i++){
        for (size_t j = 0; j < lightnum; j++){
            ray_diagram.ray[j].x += ray_diagram.ray_vel[j].x * step_size; 
            ray_diagram.ray[j].y += ray_diagram.ray_vel[j].y * step_size;
            if (ray_diagram.index[i] != 
                check_n(ray_diagram.ray[j].x, ray_diagram.ray[j].y)){
                index_p = check_n(ray_diagram.ray[j].x,
                                  ray_diagram.ray[j].y);
                theta = atan2(ray_diagram.ray_vel[j].y, 
                              ray_diagram.ray_vel[j].x);
                theta_p = asin((ray_diagram.index[j] / index_p) * sin(theta));
                ray_diagram.ray_vel[j].x = max_vel * sin(theta_p);
                ray_diagram.ray_vel[j].y = max_vel * cos(theta_p);
                ray_diagram.index[j] = index_p;
                
            }
        }
    }

    return ray_diagram;
}

