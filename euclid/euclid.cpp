/*-------------euclid.cpp-----------------------------------------------------//
*
* Purpose: Implement euclid's orchard with euclid's algorithm
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include "../visualization/cairo/cairo_vis.h"

// Function to find the greatest common divisor of two elements
int gcd(int a, int b);

// Function to create euclid's orchard
void euclid_orchard(frame &anim, int size, color on_clr, color off_clr);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    // Creating scene and frames and stuff
    int fps = 30;
    double res_x = 1000;
    double res_y = 1000;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 1};
    color on_clr = {1, 1, 1, 1};
    color off_clr = {0, 0, 0, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    euclid_orchard(layers[1], 100, on_clr, off_clr);
    draw_layers(layers);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to find the greatest common divisor of two elements, 
// Euclid's algorithm
int gcd(int a, int b){
    while (a != b){
        if (a > b){
            a = a - b;
        }
        else{
            b = b - a;
        }
    }

    return a;
}

// Function to create euclid's orchard
void euclid_orchard(frame &anim, int size, color on_clr, color off_clr){

    // First, we will animate growing the orchard
    vec loc;
    color use_clr;

    // Drawing orchard on 2d grid
    for (int i = 1; i <= size; i++){
        for (int j = 1; j <= size; j++){

            // Finding location
            loc.x = anim.res_x * 0.9 * (double)(i-1)/(size-1) 
                    + 0.05*anim.res_x;
            loc.y = anim.res_y * 0.9 * (double)(size - j)/(size-1) 
                    + 0.05*anim.res_y;

            // Setting color based on whether there is a greatest common divisor
            // greater than 1
            if (gcd(i, i+j) != 1){
                use_clr = off_clr;
            }
            else{
                use_clr = on_clr;
            }

            // grow_circle here
            //grow_square(anim, 0.1, 1, num_frames, loc, 10, use_clr);
            grow_square(anim, 0, loc, 10, use_clr);

        }
    }
}
