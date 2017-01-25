/*-------------plasma_fractal.cpp---------------------------------------------//
*
* Purpose: To use the diamond-square algorithm to create a plasma fractal
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include "../visualization/cairo/cairo_vis.h"

// Struct to hold the edges of a provided diamond or square
// The z coordinate of each vector location from vec.h will be the height
// Coordinate systems:
// Square: 0 1          diamond:   0
//         2 3                   3   1
//                                 2
struct Edges{
    std::vector<vec> edge;
};

// determine center height for diamond
void find_diamond(std::vector<vec> &texture, Edges &diamond, Edges &square);

// determine center height for square
void find_square(std::vector<vec> &texture, Edges &square, Edges &diamond);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// determine center height for diamond
void find_diamond(std::vector<vec> &texture, Edges &diamond, Edges &square){

    // Creating random variable in double space for variance on average pt
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> variance(0,1);

    // Setting average point's initial location
    vec avg = {(diamond.edge[3].x + diamond.edge[1].x) * 0.5, 
               (diamond.edge[0].y + diamond.edge[2].y) * 0.5,
               0};

    // Finding the average height of all the edges
    for (size_t i = 0; i < diamond.edge.size(); i++){
        avg.z += diamond.edge[i].z;
    }

    avg.z /= diamond.edge.size() + variance(gen);

    // Recursively create squares to work with
    Edges in_square;
    in_square.edge.reserve(4);

    // First square will be S0, D0, D3, avg
    in_square.edge[0] = square.edge[0];
    in_square.edge[1] = diamond.edge[0];
    in_square.edge[2] = diamond.edge[3];
    in_square.edge[3] = avg;

    find_square(texture, in_square, diamond);

    // Second square will be D0, S1, avg, D1
    in_square.edge[0] = diamond.edge[0];
    in_square.edge[1] = square.edge[1];
    in_square.edge[2] = avg;
    in_square.edge[3] = diamond.edge[1];

    find_square(texture, in_square, diamond);

    // Third square will be D3, avg, D2, S2
    in_square.edge[0] = diamond.edge[3];
    in_square.edge[1] = avg;
    in_square.edge[2] = diamond.edge[2];
    in_square.edge[3] = square.edge[2];

    find_square(texture, in_square, diamond);

    // Fourth square will be avg, D1, D2, S3
    in_square.edge[0] = avg;
    in_square.edge[1] = diamond.edge[1];
    in_square.edge[2] = diamond.edge[2];
    in_square.edge[3] = square.edge[3];

    find_square(texture, in_square, diamond);
    
}

// determine center height for square
void find_square(std::vector<vec> &texture, Edges &square, Edges &diamond){

    // Creating random variable in double space for variance on average pt
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> variance(0,1);

    // Setting average point's initial location
    vec avg = {(square.edge[0].x + square.edge[1].x) * 0.5, 
               (square.edge[0].y + square.edge[3].y) * 0.5,
               0};

    // Finding the average height of all the edges
    for (size_t i = 0; i < square.edge.size(); i++){
        avg.z += square.edge[i].z;
    }

    avg.z /= square.edge.size() + variance(gen);

    // Recursively create diamonds to work with
    Edges in_diamond;
    in_diamond.edge.reserve(4);

    // First diamond will be D0, S1, avg, S0
    in_diamond.edge[0] = diamond.edge[0];
    in_diamond.edge[1] = square.edge[1];
    in_diamond.edge[2] = avg;
    in_diamond.edge[3] = square.edge[0];

    find_diamond(texture, in_diamond, square);

    // Second diamond will be S1, D1, S3, avg
    in_diamond.edge[0] = square.edge[1];
    in_diamond.edge[1] = diamond.edge[1];
    in_diamond.edge[2] = square.edge[3];
    in_diamond.edge[3] = avg;

    find_diamond(texture, in_diamond, square);

    // Third diamond will be avg, S3, D2, S2
    in_diamond.edge[0] = avg;
    in_diamond.edge[1] = square.edge[3];
    in_diamond.edge[2] = diamond.edge[2];
    in_diamond.edge[3] = square.edge[2];

    find_diamond(texture, in_diamond, square);

    // Fourth diamond will be S0, avg, S2, D3
    in_diamond.edge[0] = square.edge[0];
    in_diamond.edge[1] = avg;
    in_diamond.edge[2] = square.edge[2];
    in_diamond.edge[3] = diamond.edge[3];

    find_diamond(texture, in_diamond, square);

}
