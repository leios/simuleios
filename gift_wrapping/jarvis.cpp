/*-------------jarvis.cpp-----------------------------------------------------//
* 
* Purpose: To implement a simple gift-wrapping / convex hull algorithm (jarvis)
*
*   Notes: TEST ANGLE FUNCTION!!!
*          Fix the gift_wrapping syntax
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// Struct
struct pos{
    double x, y;
};

// Struct to hold simulation data
struct parameter{
    std::vector<pos> hull, points;
};

// distance function between points
double dist(pos a, pos b){
    return sqrt((a.x - b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)) ;
}

// Finding the angle between 3 points
double angle(pos A, pos B, pos C){
    double a = dist(B,C);
    double b = dist(A,C);
    double c = dist(A,B);
    return acos((a*a - b*b - c*c)/(2*a*c));
}

// Function to initialize random points
parameter init(int num); 

// Function to wrap the points with a hull
void gift_wrap(parameter &par);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    parameter par = init(100);

    for (size_t i = 0; i < par.points.size(); i++){
        std::cout << par.points[i].x << '\t' << par.points[i].y << '\n';
    }
    std::cout << par.hull[0].x << '\t' << par.hull[0].y << '\n';
    //gift_wrap(par);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to initialize random points
parameter init(int num){
    parameter par;
    par.points.reserve(num);

    // Creating random device
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        pos_dist(0,1);

    pos tmp;
    for (int i = 0; i < num; i++){
        tmp.x = pos_dist(gen);
        tmp.y = pos_dist(gen);
        par.points.push_back(tmp);
    }

    // Find the far left point 
    std::sort(par.points.begin(), par.points.end(), 
              [](const pos a, const pos b){return a.x < b.x;});
    return par;
}

// Function to wrap the points with a hull
void gift_wrap(parameter &par){
    // expand the hull
    pos curr_hull, prev_hull, next_hull;
    prev_hull.x = 0.0;
    prev_hull.y = 0.0;
    double threshold = 0.001, angle_tmp, final_angle;
    int i = 0;
    while (dist(curr_hull, par.hull[0]) > threshold){
        if (i == 0){
            curr_hull = par.hull[0];
        }

        final_angle = 2*M_PI;
        for (int j = 0; j < par.points.size(); j++){
             angle_tmp = angle(prev_hull, curr_hull, par.points[j]);
             if (angle_tmp < final_angle){
                 final_angle = angle_tmp;
                 next_hull = par.points[j];
             }
        }
        par.hull.push_back(next_hull);
        prev_hull = curr_hull;
        curr_hull = next_hull;
    }
}
