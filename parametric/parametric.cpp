/*-------------parametric.cpp-------------------------------------------------//
*
* Purpose: Visualize simple parametric equations
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include "../visualization/cairo/cairo_vis.h"

// Function to draw a circle
std::vector<vec> create_circle(int res);

// Set of functions to draw a heart
std::vector<vec> create_heart(int res, int dist);

// Function to draw
void draw(std::vector<vec> &array);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initialize all visualization components
    //std::vector<vec> circle = create_circle(100);
    //draw(circle);
    std::vector<vec> heart = create_heart(100,1);
    draw(heart);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to draw a circle
std::vector<vec> create_circle(int res){

    std::vector<vec> circle(res);

    // Parametric equation for a circle
    double theta = 0.0;
    for (int i = 0; i < res; ++i){
        theta = -M_PI + 2.0*M_PI*i/(res-1);
        circle[i].x = cos(theta);
        circle[i].y = sin(theta);
    }

    return circle;
}

// Set of functions to draw a heart
std::vector<vec> create_heart(int res, int dist){

    std::vector<vec> heart(res);

    double theta;

    for (int i = 0; i < res; i++){
        theta = -M_PI + 2*M_PI*i/(res-1);
        switch(dist){
            case 0:
            {
                heart[i].x = cos(theta);
                heart[i].y = sin(theta);
                break;
            }
            case 1:
            {
                heart[i].x = pow(sin(theta), 3);
                heart[i].y = 13*cos(theta) - 5 * cos(2 * theta) 
                             -2*cos(3*theta) - cos(4*theta);
                break;
            }
        }
    }

    return heart;
}

// Function to draw
void draw(std::vector<vec> &array){
    // Initialize all visualization components
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 0};
    color line_clr = {1, 0, 0, 1};

    std::vector<frame> layers = init_layers(1, res, fps, bg_clr);

    // shift the array to be in range 0->1 from -1->1

    vec maximum = {1, 1};
    vec minimum = {-1, -1};
    vec temp;

    temp = *std::max_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.x < j.x;});
    maximum.x = temp.x;
    temp = *std::max_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.y < j.y;});
    maximum.y = temp.y;

    temp = *std::min_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.x < j.x;});
    minimum.x = temp.x;
    temp = *std::min_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.y < j.y;});
    minimum.y = temp.y;

    std::cout << maximum.x << '\t' << maximum.y << '\n';
    std::cout << minimum.x << '\t' << minimum.y << '\n';


    for (int i = 0; i < array.size(); i++){
        array[i].x = (array[i].x - minimum.x)/ (maximum.x - minimum.x);
        array[i].y = (array[i].y - minimum.y)/ (maximum.y - minimum.y);
    }

    draw_array(layers[0], array, res_x, res_y, line_clr);

    draw_layers(layers);

    for (int i = 0; i < layers.size(); ++i){
        layers[i].destroy_all();
    }

}

