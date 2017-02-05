/*-------------definitely_a_date.cpp------------------------------------------//
*
* Purpose: To draw a heart using parametric equations
*
*   Notes: - I use vi, get over it
*          - I don't normally use this font, but it was the only one that 
*              looked decent when on a larger screen
*          - Here are 3 questions you should ask every date:
*              - Tabs or spaces?
*              - Vim or emacs?
*              - What is your preferred programming language?
*          - If they answer in any of the following ways, ^C!
*              - I don't care / Why does it matter? / I had a TABby cat once...
*              - What's an emac? / vi 
*              - FORTRAN (probably too old) / scratch (probably too young)
*          - The drawing library used here is a little too complicated for
*              my liking and I will probably start writing a new one soon
*              for future episodes
*-----------------------------------------------------------------------------*/

// including necessary directories
#include <iostream>

// vectors make your life easier, but arrays will work too!
#include <vector>

// I wrote this one myself, sorry!
#include "../visualization/cairo/cairo_vis.h"

// Alright, let's get to drawing some shapes, woo!
int main(){

    // Defining necessary variables
    int res = 100;
    double r = 1;
    double theta = 0;

    // We will be using a vector of size 100 for our shape
    std::vector<vec> shape(100);

    // Now for a loop to define the shape! (pre-increment for optimal performance)
    for (int i = 0; i < res; ++i){

        // theta will go from 0->2pi
        theta = 2 * M_PI * i / (res-1);

        // Now to define the actual shape!
        shape[i].y = r * cos(theta) + cos(2*theta);
        shape[i].x = pow(r * sin(theta), 3);
    }

    // This function comes from my drawing library, sorry!
    draw(shape);
}
