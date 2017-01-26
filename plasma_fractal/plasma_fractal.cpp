/*-------------plasma_fractal.cpp---------------------------------------------//
*
* Purpose: To use the diamond-square algorithm to create a plasma fractal
*
*   Notes: Change edges to be a vector of vec (positions), and use 
*              vec_to_int to update texture
*          The initial diamond is smaller than it needs to be
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
    std::vector<int> edge;
};

// determine center height for diamond
void find_diamond(std::vector<double> &texture, Edges &diamond, Edges &square,
                  int count);

// determine center height for square
void find_square(std::vector<double> &texture, Edges &square, Edges &diamond,
                 int count);

// Function to tranform from an integer to a vector location for a grid size
vec int_to_vec(int id, int size);

// Function to transform a vector to an integer for a grid
int vec_to_int(vec loc, int size);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Creating scene / frames and stuff
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0,0,0,1};
    color line_clr = {1,1,1,1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    // Creating initial texture of all 1's
    int size = 5;
    std::vector<double> texture(size*size);
    for (int i = 0; i < size*size; i++){
        texture[i] = 1;
    }

    Edges square, diamond;

    square.edge.reserve(4);
    diamond.edge.reserve(4);

    // Setting initial diamond and square
    square.edge[0] = 0;
    square.edge[1] = size*(size-1);
    square.edge[2] = size-1;
    square.edge[3] = size*size-1;

    diamond.edge[0] = size * (size-1) * 0.5;
    diamond.edge[1] = size * (size-1) + (size - 1) * 0.5;
    diamond.edge[2] = size * (size-1) * 0.5 + size - 1;
    diamond.edge[3] = (size-1) * 0.5;

    vec d_loc, s_loc;
    for (int i = 0; i < 4; i++){
        d_loc = int_to_vec(diamond.edge[i], size);
        s_loc = int_to_vec(square.edge[i], size);

        d_loc.x = (d_loc.x / size) * res.x + res.x / (2*size);
        d_loc.y = ((size - d_loc.y) / size) * res.y - res.y / (2*size);

        s_loc.x = (s_loc.x / size) * res.x + res.x / (2*size);
        s_loc.y = ((size - s_loc.y) / size) * res.y - res.y / (2*size);

        //grow_circle(layers[1], 1, d_loc, 10, line_clr);
        //grow_circle(layers[1], 1, s_loc, 10, line_clr);
    }

    find_square(texture, square, diamond, 4);

    std::cout << "recursion finished" << '\n';

    double max = *max_element(texture.begin(), texture.end());
    color pt_clr;
    vec pt_loc;
    for (int i = 0; i < texture.size(); i++){
        std::cout << texture[i] << '\n';
        texture[i] /= max;
        pt_clr = {1-texture[i],0,texture[i],1};
        pt_loc = int_to_vec(i, size);
        pt_loc.x = (pt_loc.x / size) * res.x + res.x / (2*size);
        pt_loc.y = ((size - pt_loc.y) / size) * res.y - res.y / (2*size);

        grow_circle(layers[1], 0, pt_loc, 10, pt_clr);
    }

    draw_layers(layers);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// determine center height for diamond
void find_diamond(std::vector<double> &texture, Edges &diamond, Edges &square,
                  int count){

    count -= 1;

    if (count >= 0){

        // Creating random variable in double space for variance on average pt
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> variance(0,1);
    
        // Setting average point's initial value
        double avg = 0.0;
    
        // Finding the location of the average point
        int size = sqrt(texture.size());
        vec avg_loc = {(int_to_vec(diamond.edge[3], size).x
                       + int_to_vec(diamond.edge[1], size).x) * 0.5,
                       (int_to_vec(diamond.edge[1], size).y
                       + int_to_vec(diamond.edge[2], size).y) * 0.5};
    
        int avg_id = vec_to_int(avg_loc, size);
    
        // Finding the average height of all the edges
        for (size_t i = 0; i < diamond.edge.size(); i++){
            avg += texture[diamond.edge[i]];
        }
    
        avg /= (double)diamond.edge.size() + variance(gen);
    
        // Updating texture
        texture[avg_id] = avg;
    
        // Recursively create squares to work with
        Edges in_square;
        in_square.edge.reserve(4);
        for (int i = 0; i < 4; i++){
            in_square.edge.push_back(0);
        }
    
        // First square will be S0, D0, D3, avg
        in_square.edge[0] = square.edge[0];
        in_square.edge[1] = diamond.edge[0];
        in_square.edge[2] = diamond.edge[3];
        in_square.edge[3] = avg_id;
    
        find_square(texture, in_square, diamond, count);
    
        // Second square will be D0, S1, avg, D1
        in_square.edge[0] = diamond.edge[0];
        in_square.edge[1] = square.edge[1];
        in_square.edge[2] = avg_id;
        in_square.edge[3] = diamond.edge[1];
    
        find_square(texture, in_square, diamond, count);
    
        // Third square will be D3, avg, D2, S2
        in_square.edge[0] = diamond.edge[3];
        in_square.edge[1] = avg_id;
        in_square.edge[2] = diamond.edge[2];
        in_square.edge[3] = square.edge[2];
    
        find_square(texture, in_square, diamond, count);
    
        // Fourth square will be avg, D1, D2, S3
        in_square.edge[0] = avg_id;
        in_square.edge[1] = diamond.edge[1];
        in_square.edge[2] = diamond.edge[2];
        in_square.edge[3] = square.edge[3];

        find_square(texture, in_square, diamond, count);
    }
    
}

// determine center height for square
void find_square(std::vector<double> &texture, Edges &square, Edges &diamond, 
                 int count){

    count -= 1;

    if (count >= 0){

        // Creating random variable in double space for variance on average pt
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> variance(0,1);
    
        // Setting average point's initial value
        double avg = 0.0;
    
        // Finding the location of the average point
        int size = sqrt(texture.size());
        vec avg_loc = {(int_to_vec(square.edge[0], size).x
                       + int_to_vec(square.edge[1], size).x) * 0.5,
                       (int_to_vec(square.edge[0], size).y
                       + int_to_vec(square.edge[2], size).y) * 0.5};
    
        int avg_id = vec_to_int(avg_loc, size);
    
    
        // Finding the average height of all the edges
        for (size_t i = 0; i < square.edge.size(); i++){
            avg += texture[square.edge[i]];
        }
    
        avg /= (double)square.edge.size() + variance(gen);
    
        texture[avg_id] = avg;
    
        // Recursively create diamonds to work with
        Edges in_diamond;
        in_diamond.edge.reserve(4);
        for (int i = 0; i < 4; i++){
            in_diamond.edge.push_back(0);
        }
    
        // First diamond will be D0, S1, avg, S0
        in_diamond.edge[0] = diamond.edge[0];
        in_diamond.edge[1] = square.edge[1];
        in_diamond.edge[2] = avg_id;
        in_diamond.edge[3] = square.edge[0];
    
        find_diamond(texture, in_diamond, square, count);
    
        // Second diamond will be S1, D1, S3, avg
        in_diamond.edge[0] = square.edge[1];
        in_diamond.edge[1] = diamond.edge[1];
        in_diamond.edge[2] = square.edge[3];
        in_diamond.edge[3] = avg_id;
    
        find_diamond(texture, in_diamond, square, count);
    
        // Third diamond will be avg, S3, D2, S2
        in_diamond.edge[0] = avg_id;
        in_diamond.edge[1] = square.edge[3];
        in_diamond.edge[2] = diamond.edge[2];
        in_diamond.edge[3] = square.edge[2];
    
        find_diamond(texture, in_diamond, square, count);
    
        // Fourth diamond will be S0, avg, S2, D3
        in_diamond.edge[0] = square.edge[0];
        in_diamond.edge[1] = avg_id;
        in_diamond.edge[2] = square.edge[2];
        in_diamond.edge[3] = diamond.edge[3];
    
        find_diamond(texture, in_diamond, square, count);
    }

}

// Function to tranform from an integer to a vector location for a grid size
vec int_to_vec(int id, int size){

    vec loc = {floor(id/size), id % size};
    //std::cout << id << '\t' << loc.x << '\t' << loc.y << '\n';
    return loc;
}

// Function to transform a vector to an integer for a grid
int vec_to_int(vec loc, int size){

    return loc.y + loc.x*size;
}
