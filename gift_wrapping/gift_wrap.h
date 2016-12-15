/*-------------gift_wrap.h----------------------------------------------------//
*
* Purpose: Header file for jarvis.cpp
*
*-----------------------------------------------------------------------------*/
#ifndef GIFT_WRAP_H
#define GIFT_WRAP_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../visualization/cairo/cairo_vis.h"

// Struct to hold simulation data
struct parameter{
    std::vector<vec> hull, points;
};

// distance function between points
double dist(vec a, vec b){
    return sqrt((a.x - b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)) ;
}

// Finding the angle between 3 points
double angle(vec A, vec B, vec C){
    double a = dist(B,C);
    double b = dist(A,C);
    double c = dist(A,B);
    return acos((b*b - a*a - c*c)/(2*a*c));
}

// Function to test the angle function
void test_angle();

// Function to initialize random points
parameter init(int num); 

// Function to wrap the points with a hull
void gift_wrap(parameter &par, std::vector<frame> &anim);

#endif
