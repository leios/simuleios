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
    vec res = {400, 300};
    int fps = 30;
    color bg_clr = {0,0,0,1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);
/*
    std::vector<frame> layers(3);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(400, 300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    //create_bg(layers[0], 0, 0, 0);
*/

    // Defining color to use for lines
    color line_clr = {1, 1, 1, 1};
    color line_clr_2 = {1, 0, 1, 1};

    vec ori = {200, 150};

    //grow_circle(layers[1], 1, 100, 200, ori, 10, 1); 

    vec dim = {100, 150};
    grow_rect(layers[1], 1, 100, 200, ori, dim, line_clr);
    grow_rect(layers[1], 1, ori, dim, line_clr_2);
    grow_square(layers[1], 1, 200, 300, ori, 200, line_clr);
    grow_circle(layers[1], 1, ori, 200, line_clr_2);
/*

    std::vector<int> array(100);
    for (int i = 0; i < 100; i++){
        array[i] = i;
    }

    std::vector<std::vector<int>> perms(12);
    for (int i = 0; i < perms.size(); i++){
        perms[i] = array;
    }

    draw_permutations(layers[1], 0, 50, perms, ori, 400, 300, 
                      line_clr, line_clr);

    //bar_graph(layers[1], 0, 50, 100, array, ori, 200, 150, line_clr);

    color highlight_clr = {1, 0, 0, 1};

    //highlight_bar(layers[1], 51, 80, array, ori, 200, 150, highlight_clr, 5);
*/

    draw_layers(layers);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/
