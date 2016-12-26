/*-------------vis_test.cpp---------------------------------------------------//
* 
* Purpose: To test visualizing our vairo animations
*
*-----------------------------------------------------------------------------*/

#include "cairo_vis.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initialize visualization stuff
    std::vector<frame> layers(3);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(400, 300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], 0, 0, 0);

    // Defining color to use for lines
    color line_clr = {1, 1, 1, 1};

    vec ori = {50, 50};

    //grow_circle(layers[1], 1, 100, 200, ori, 10, 1); 

    std::vector<int> array(100);
    for (int i = 0; i < 100; i++){
        array[i] = i;
    }

    bar_graph(layers[1], 10, array, 400, 300, line_clr);

    draw_layers(layers);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/
