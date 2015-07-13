/*-------------bassdrop.cpp---------------------------------------------------//
*
*              bassdrop 
*
* Purpose: Check README. We are droppin' the bass, and checking to see how long
*          it takes to fall!
* 
*   Notes: x = x_0 + v_0 * t + (0.5) * a * t * t
*          v  = v_0 + a * t
*
*-----------------------------------------------------------------------------*/

#include<iostream>

using namespace std;

double simple(double timestep, double height);
double simple2(double timestep, double height);
double rk4(double timestep, double height);

int main(){

    double timestep  = 0.01, height = 5;
    double time1 = simple(timestep, height);
    double time2 = simple2(timestep, height);
    double rk4_time = rk4(timestep, height);

    cout << "The time it takes to hit the ground is:" << endl;
    cout << time1 << " with the first algorithm"<< endl;
    cout << time2 << " with the second algorithm" << endl;
    cout << rk4_time << " with the rk4 algorithm" << endl;

    return 0;
}

// Finds the time with a simple algorithm
double simple(double timestep, double height){

    double time = 0, x = height;
    while (x > 0){
        time += timestep;
        x = height - (0.5) * 10 * time * time;
    }

    return time;
}

// Finds the tiem with another simple algorithm
double simple2(double timestep, double height){
    double time = 0, x = height, vel = 0;

    while (x > 0){
        x += vel * timestep - (0.5) * 10 * timestep * timestep;
        time += timestep;
        vel += -10 * timestep;
    }

    return time;
}

// Finds the time with the Runge-Kutta method! Woo!
double rk4(double timestep, double height){
    double time = 0, k1, k2, k3, k4;
    double x = height, vel =0, g = -10;

    while (x > 0){

        k1 = g * time;
        k2 = g * (time +(timestep * 0.5));
        k3 = g * (time +(timestep * 0.5));
        k3 = g * (time + timestep);

        x += (timestep / 6.0) * (k1 + 2 * (k2 +k3) + k4);
        time += timestep;

    }
    return time;
}

