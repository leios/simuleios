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
    color wrap_clr = {0, 0, 1, 1};
    color wrap_clr2 = {1, 1, 1, 1};
    bool chan = false;
};

// distance function between points
double dist(vec a, vec b){
    return sqrt((a.x - b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)) ;
}

// Function to find angle if casting along origin
double cast_angle(vec v);

// Finding the angle between 3 points
double angle(vec A, vec B, vec C);

// function to draw an array (or vector of vec's)
void draw_array(frame &anim, std::vector<vec> &array,
                double x_range, double y_range, color wrap_clr);

// Function to test the angle function
void test_angle();

// Function to draw random distribution with grow_circle command
void grow_dist(parameter &par, std::vector<frame> &layers, 
               double x_range, double y_range);

// Function to initialize random points
parameter init(int num); 

// Function to find CCW rotation
double ccw(vec a, vec b, vec c);

// test function for ccw
void ccw_test();

// Function to wrap the points with a hull
void jarvis(parameter &par, std::vector<frame> &layers);

// Function to wrap points in hull (GRAHAM SCAN EDITION)
void graham(parameter &par, std::vector<frame>& layers);

// Function for Chan's algorithm
void chan(parameter &par, int subhull, std::vector<frame>& layers);

#endif
