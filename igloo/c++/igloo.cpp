/*-----------igloo.cpp--------------------------------------------------------//
*
*            igloo.cpp
*
* Purpose: To answer fun questions with code.
*
* Concept: general concept provided in ../README. Be sure to read it, it's cool.
*
*   Notes: This is the first file I have written to solve this problem. I may
*          choose to solve it again with other programming languages in the 
*          near future, but for now... c++.
*
*-----------------------------------------------------------------------------*/

#include<iostream>

using namespace std;

double integrate_sphere();

int main(){

    double variable = integrate_sphere();
    cout << "the final integral is: " << variable << "! Woo!" << endl;
}

// This function will integrate a half-sphere.
double integrate_sphere(){

    int count = 0, res = 1000;
    double integral, x, y, z;
    for (int ix = 0; ix < res; ix++){
        for (int iy = 0; iy < res; iy++){
            for (int iz = 0; iz < res / 2; iz++){
                x = (2 * ix / res) - 1; 
                y = (2 * iy / res) - 1;
                z = (2 * iz / res);
                if (((x*x) + (y*y) + (z*z)) < 1){
                    count += 1;
                }
            }
        }
    }

    integral = (count * (2 / 1000.0) * (2 / 1000.0) * (2 / 500.0));

    return integral;
}
