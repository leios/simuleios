/*------------optics_vis.h----------------------------------------------------//
*
* Purpose: Header file for optics_vis.cpp, holds all functions and structures
*
*   Notes: This will be using Cairo, be careful
*
*-----------------------------------------------------------------------------*/

#ifndef OPTICS_VIS_H
#define OPTICS_VIS_H

#include <cairo.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
#include <sstream>
#include <vector>

//#define num_frames 300
#define num_frames 300

// A very simple vector type with operators that are used in this file
struct vec {
    double x, y;

    vec() : x(0.0), y(0.0) {}
    vec(double x0, double y0) : x(x0), y(y0) {}
};


// Struct for colors
struct color{
    double r, g, b;
};

// Struct to hold all the necessary data for animations
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

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight);

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time, 
                  vec &ori_1, vec &ori_2, color &clr);

// Function to draw layers
void draw_layers(std::vector<frame> &layer);

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, vec ori, 
                    color clr);

// Function to draw lens at provided position
void draw_lens(std::vector<frame> &layer, double time, 
               const struct sphere &lens);

// function to create vector<double> for index_plot function
std::vector<double> create_index_texture(const sphere &lens);

// function to fill inside of lens with appropriate refractive index colors
void index_plot(frame &anim, int framenum,
                const sphere &lens, color lens_clr, double max_alpha);

// overloaded function to fill inside of lens with appropriate ior colors
void index_plot(frame &anim, int framenum, 
                std::vector<double> &index_texture, 
                const sphere &lens, color lens_clr, double max_alpha);

// overloaded function to fill inside of lens with appropriate ior colors
void index_plot(frame &anim, int framenum, 
                std::vector<unsigned char> &index_texture, 
                const sphere &lens, color lens_clr, double max_alpha);


#endif 
