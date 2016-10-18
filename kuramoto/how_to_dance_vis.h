/*------------how_to_dance.h--------------------------------------------------//
*
* Purpose: to visualize Steve's awkward dancing
*
*-----------------------------------------------------------------------------*/

#ifndef HOW_TO_DANCE_VIS
#define HOW_TO_DANCE_VIS

//#include "how_to_dance.h"
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
void draw_human(frame &anim, double pos, double phase, double freq, color clr);

#endif
