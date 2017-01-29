/*-------------relationship.cpp-----------------------------------------------//
*
* Purpose: to model becoming an ideal relationship by shape matching and 
*          modification using shape contexts. Info here:
*              https://en.wikipedia.org/wiki/Shape_context
*
*   Notes: Ideal relationship shape will be a heart (aww)
*          
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <stdlib.h>
#include "../visualization/cairo/cairo_vis.h"

// Function to create ideal relationship shape (heart)
std::vector<vec> create_ideal_shape(int res);

// Function to return a random blob
std::vector<vec> create_blob(int res);

// Function to visualize the addition of two blobs
std::vector<vec> add_blob(std::vector<vec> blob_1, std::vector<vec> blob_2);

// Shape matching functions, random sampling is unnecessary here

// Function to generate shape contexts
// In principle, the shape contexts should be a vector of vector<double>
std::vector<double> gen_shape_context(std::vector<vec> shape);

// Function to align shapes
std::vector<vec> align_shapes(std::vector<vec> shape_1, 
                              std::vector<vec> shape_2);

// Function to compute the shape distance
std::vector<double> gen_shape_dist(std::vector<double> context_1,
                                   std::vector<double> context_2);

// Function to modify the shape (relationship) to match the perfect shape
std::vector<vec> match_shape(std::vector<vec> shape_1, 
                             std::vector<vec> shape_2);


/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initialize all visualization components
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 0};
    color line_clr = {1, 1, 1, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    std::vector<vec> heart = create_ideal_shape(100);
    std::vector<vec> blob = create_blob(100);

    for (int i = 0; i < heart.size(); i++){
        std::cout << blob[i].x << '\t' << heart[i].y << '\n';
    }

    draw_array(layers[1], blob, 400, 400, line_clr);

    draw_layers(layers);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create ideal relationship shape (heart)
std::vector<vec> create_ideal_shape(int res){
    // Creating the vector to work with
    std::vector<vec> ideal_shape(res);

    double t = 0;
    for (int i = 0; i < res; i++){
        t = -M_PI + 2.0*M_PI*i/res;
        ideal_shape[i].x = 16 * sin(t)*sin(t)*sin(t) / 34.0 + 0.5;
        ideal_shape[i].y = (13*cos(t) - 5*cos(2*t) -2*cos(3*t)-cos(4*t))/34.0
                            +0.5;
    }

    return ideal_shape;
}


// Function to return a random blob
std::vector<vec> create_blob(int res){

    // Creating vector to work with
    std::vector<vec> blob(res);

    // Doing random things
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> variance(0,1);

    double t = 0;
    for (int i = 0; i < res; i++){
        t = 2.0*M_PI*i/res;
        //blob[i].x = cos(t)*(0.4+variance(gen)*0.1) + 0.5;
        //blob[i].y = sin(t)*(0.4+variance(gen)*0.1) + 0.5;
        blob[i].x = (15+variance(gen)) * sin(t)*sin(t)*sin(t) / 34.0 + 0.5;
        blob[i].y = ((12+variance(gen))*cos(t) 
                      -(4+variance(gen))*cos(2*t) 
                      -(1+variance(gen))*cos(3*t)
                      -(0.9 + variance(gen)*0.1)*cos(4*t))/34.0
                            +0.5;

    }

    return blob;
}

// Function to visualize the addition of two blobs
// Might change depending on the underlying blob shape (could be a heart)
std::vector<vec> add_blob(std::vector<vec> blob_1, std::vector<vec> blob_2){

    std::vector<vec> blob_sum = blob_1;

    // For now, we are taking whichever points of the two blobs is closest to
    // a circle
    double t;
    vec close_1, close_2;
    for (size_t i = 0; i < blob_1.size(); i++){
        t = 2.0*M_PI*i/blob_1.size();
        close_1.x = abs(blob_1[i].x - (cos(t)*0.5 + 0.5));
        close_1.y = abs(blob_1[i].y - (sin(t)*0.5 + 0.5));
        close_2.x = abs(blob_2[i].x - (cos(t)*0.5 + 0.5));
        close_2.y = abs(blob_2[i].y - (sin(t)*0.5 + 0.5));

        if (length(close_2) < length(close_1)){
            blob_sum[i] = blob_2[i];
        }
    }

    return blob_sum;

}
