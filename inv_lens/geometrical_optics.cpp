/*-------------geometrical_optics.cpp-----------------------------------------//
*
*              geometrical optics
*
* Purpose: to simulate light going through a lens with a variable refractive 
*          index. Not wave-like.
*
*   Notes: Compiles with g++ geometrical_optics.cpp -std=c++11
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
double check_n(double x, double y, dimensions origin, double radius);

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double angle, 
                     dimensions origin, double radius);

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel,
                     dimensions origin, double radius,
                     std::ofstream &output);

// Finds the c for later
double find_c(double ix, double iy, dimensions origin, double radius);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defines output
    std::ofstream output("out.dat", std::ofstream::out);

    dimensions dim;
    dim.x = 4;
    dim.y = 10;

    // origin of the sphere
    dimensions origin;
    origin.x = 5;
    origin.y = 5;

    double radius = 5;

    double max_vel = 1;
    light_rays pvec = light_gen(dim, max_vel, 0.523598776, origin, radius);

    pvec = propagate(pvec, 0.1, max_vel, origin, radius, output);

    output << "\n \n5	0	0	0 \n5	5	0	0 \n";

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// checks refractive index profile
double check_n(double x, double y, dimensions origin, double radius){
    // check refractive index against a set profile

    double index, diff;

    diff = sqrt((x - origin.x) * (x - origin.x) 
                + (y - origin.y) * (y - origin.y));

    if (diff < radius){
        index = 1.4;
    }
    else{
        index = 1;
    }

/*
    if (x < 5 || x > 7){
        index = 1.0;
    }
    else{
        //index = (x - 4.0);
        index = 1.4;
    }
*/

    return index;
}

// Generate light and refractive index profile
light_rays light_gen(dimensions dim, double max_vel, double angle, 
                     dimensions origin, double radius){

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
                                       ray_diagram.ray[i].y, origin, radius);
    }

    return ray_diagram;
}

// Propagate light through media
light_rays propagate(light_rays ray_diagram, double step_size, double max_vel,
                     dimensions origin, double radius,
                     std::ofstream &output){

    double index_p, theta, theta_p;
    double iratio, dotprod;

    // move simulation every timestep
    for (size_t i = 0; i < time_res; i++){
        for (size_t j = 0; j < lightnum; j++){
            ray_diagram.ray[j].x += ray_diagram.ray_vel[j].x * step_size; 
            ray_diagram.ray[j].y += ray_diagram.ray_vel[j].y * step_size;
            if (ray_diagram.index[j] != 
                check_n(ray_diagram.ray[j].x, ray_diagram.ray[j].y, 
                        origin, radius)){
                index_p = check_n(ray_diagram.ray[j].x,
                                  ray_diagram.ray[j].y, origin, radius);

                std::cout << index_p << '\t' << i << '\t' << j << '\n';

/*
                // Non vector form
                theta = atan2(ray_diagram.ray_vel[j].y, 
                              ray_diagram.ray_vel[j].x);
                theta_p = asin((ray_diagram.index[j] / index_p) * sin(theta));
                ray_diagram.ray_vel[j].y = max_vel * sin(theta_p);
                ray_diagram.ray_vel[j].x = max_vel * cos(theta_p);
*/

                // Vector form -- Solution by Gustorn!
                double r = ray_diagram.index[j] / index_p;
                double mag = std::sqrt(ray_diagram.ray_vel[j].x * 
                                       ray_diagram.ray_vel[j].x +
                                       ray_diagram.ray_vel[j].y * 
                                       ray_diagram.ray_vel[j].y); 


                double ix = ray_diagram.ray_vel[j].x / mag;
                double iy = ray_diagram.ray_vel[j].y / mag;

                // c for later; Normal was: [-1, 0]
                double c = find_c(ix, iy, origin, radius);

                double k = 1.0 - r * r * (1.0 - c * c);

                if (k < 0.0) {
                    // Do whatever
                } else {
                    double k1 = std::sqrt(k);
                    ray_diagram.ray_vel[j].x = r * ix - (r * c - k1);
                    ray_diagram.ray_vel[j].y = r * iy;
                }
                ray_diagram.index[j] = index_p;
            }

        }

/*
        output << ray_diagram.ray[5].x <<'\t'<< ray_diagram.ray[5].y << '\t'
               << ray_diagram.ray_vel[5].x <<'\t'<< ray_diagram.ray_vel[5].y
               << '\n';
*/

        for (size_t q = 0; q < lightnum; q++){
            output << ray_diagram.ray[q].x <<'\t'<< ray_diagram.ray[q].y << '\t'
                   << ray_diagram.ray_vel[q].x <<'\t'<< ray_diagram.ray_vel[q].y
                   << '\n';
        }

        output << '\n' << '\n';
    }

    return ray_diagram;
}

// Finds the c for later
double find_c(double ix, double iy, dimensions origin, double radius){

    // Step 1: define normal vector
    // In this case, the normal vector is just the direction from the radius
    // of the sphere
    double mag = sqrt((ix - origin.x) * (ix - origin.x)
                      + (iy - origin.y) * (iy - origin.y));
    double x, y;
    x = (ix - origin.x) / mag;
    y = (iy - origin.y) / mag;

    // Step 2: simple dot product
    double dot = -(ix * x + iy * y);

    return dot;
}

