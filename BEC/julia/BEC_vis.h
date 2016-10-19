/*------------BEC_vis.h-------------------------------------------------------//
*
* Purpose: to visualize becoming a BEC in 1d
*
*-----------------------------------------------------------------------------*/

#ifndef BEC_VIS_H
#define BEC_VIS_H

#include <iostream>
#include <iomanip>
#include <cairo.h>
#include <sstream>
#include <math.h>

#define num_frames 500

// Struct for colors
struct color{
    double r, g, b, a;
};

// struct for x and y on the cairo grid
struct vec{
    double x, y;
};

struct frame{
    int res_x, res_y;
    int fps;
    int curr_frame;
    cairo_surface_t *frame_surface[num_frames];
    cairo_t *frame_ctx[num_frames];
    cairo_surface_t *bg_surface;
    cairo_t *bg_ctx;
    vec origin;
    std::string pngbase;

    // Function to call frame struct
    void create_frame(int x, int y, int ps, std::string pngname);

    // Function to initialize the frame struct
    void init();

    // Function to draw all frames in the frame struct
    void draw_frames();

    // Function to destroy all contexts and surfaces
    void destroy_all();

};

// Function to create basic colored background
void create_bg(frame &anim, int r, int g, int b);

// Function to plot sinunoidal 
void draw_BEC(frame &anim, std::ofstream &file);

#endif
