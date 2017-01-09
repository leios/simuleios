/*-------------complexity.cpp-------------------------------------------------//
*
* Purpose: to visualize different complexity cases (n, n^2, log(n), n!, 1)
*
*   Notes: Change complexity visualizations to read in animation layers
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include "../visualization/cairo/cairo_vis.h"

// Visualizing complexity of n
void complexity_n(int n);

// complexity of n^2
void complexity_n2(int n);

// complexity log(n)
void complexity_logn(int n);

// complexity n!
void complexity_factorial(int n);

// complexity 1
void complexity_1(int n);

// function to plot all complexity cases
void plot_complexity(int elements);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    //complexity_n2(8);
    plot_complexity(100);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// complexity of n
void complexity_n(int n){

    // Creating scene and frames and stuff
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 0};
    color square_clr = {1, 1, 1, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    double dx = (res_x / n);
    vec pos = {dx * 0.5, res_y * 0.5};

    // Let's start with a single line of boxes.
    for (int i = 0; i < n; i++){

        grow_square(layers[0], 0.5, pos, 30, square_clr);

        //defining new x positions
        pos.x += dx;
        
    }

    draw_layers(layers);
}

// complexity of n^2
void complexity_n2(int n){

    // Creating scene and frames, etc...
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 0};
    color square_clr = {1, 1, 1, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    double dx = (res_x / n);
    vec pos = {dx * 0.5, res_y - dx*0.5};

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            grow_square(layers[0], 0.5, pos, 30, square_clr);
            pos.x += dx;
        }
        pos.x = dx * 0.5;
        pos.y -= dx;
    }

    draw_layers(layers);
}

// Function to plot all of the complexity cases vs n
void plot_complexity(int elements){

    // Creating scene and frames, etc...
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 1};
    color axis_clr = {1, 1, 1, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    vec ori = {0.05*res_x, res_y - 0.05*res_y};
    vec top = {0.05*res_x,0.05*res_y};
    vec bot = {res_x - 0.05 * res_x, res_y - 0.05*res_y};
    // Drawing the axis
    animate_line(layers[1], 1, 1, ori, top, axis_clr);
    animate_line(layers[1], 1, 1, ori, bot, axis_clr);

    // Creating an array for each complexity case and drawing that array;
    std::vector<vec> complexity(elements);

    // For linear case O(n)
    for (int i = 0; i < elements; i++){
        complexity[i].x = (1.0 / elements) * i;
        complexity[i].y = (1.0 / elements) * i;
    }

    draw_array(layers[1], 0, complexity, res_x, res_y, axis_clr);

    // For n^2 case
    for (int i = 0; i < elements; i++){
        complexity[i].x = (10.0 / elements) * i;
        complexity[i].y = complexity[i].x * complexity[i].x;

        complexity[i].x /= 10.0;
        complexity[i].y /= 10.0;
    }

    draw_array(layers[1], 0, complexity, res_x, res_y, axis_clr);

    // For log(n) case
    for (int i = 0; i < elements; i++){
        complexity[i].x = (10.0 / elements) * i;
        complexity[i].y = log2(complexity[i].x);

        if (complexity[i].y < 0){
            complexity[i].y = 0;
        }

        complexity[i].x /= 10.0;
        complexity[i].y /= 10.0;

    }

    draw_array(layers[1], 0, complexity, res_x, res_y, axis_clr);

    draw_layers(layers);
}
