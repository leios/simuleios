/*------------cairo_vis.h-----------------------------------------------------//
*
* Purpose: to visualize 2d simulation data for simuleios
*
*-----------------------------------------------------------------------------*/

#ifndef HOW_TO_DANCE_VIS
#define HOW_TO_DANCE_VIS

#include <iostream>
#include <iomanip>
#include <cairo.h>
#include <sstream>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>
#include "vec.h"

#define num_frames 1000

// Struct for colors
struct color{
    double r, g, b, a;
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
void create_bg(frame &anim, double r, double g, double b);
void create_bg(frame &anim, double r, double g, double b, double a);

// Creating basic colored background
void color_bg(frame &anim, int start_layer, int r, int g, int b);

// Function to draw human stuff
void draw_human(frame &anim, double pos, double phase, double freq, color clr);
void draw_human(frame &anim, vec pos, double height, color clr);
void animate_human(frame &anim, vec pos, double height, color clr, 
                   int start_frame, int end_frame, double time);
void animate_human(frame &anim, vec pos, double height, color clr, double time);
void write_text(frame &anim, vec head_pos, vec text_pos, double head_radius,
                double font_size, std::string text);

void write_fraction(frame &anim, vec frac_pos, int num, int den,
                    double font_size, color font_color);

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, double weight);
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight);
void grow_circle(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, color cir_clr);
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 color cir_clr);

//// Function to draw a filled circle
void draw_filled_circle(frame &anim, vec ori, double radius, int draw_frame,
                        color cir_clr);

// Function to grow a square at the provided point.
void grow_square(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, color square_clr);
void grow_square(frame &anim, double time,
                 vec &ori, double radius, color square_clr);

// Functions to grow rectangles
void grow_rect(frame &anim, double time, int start_frame, int end_frame,
               vec &ori, vec &dim, color rect_clr);
void grow_rect(frame &anim, double time,
               vec &ori, vec &dim, color rect_clr);

// Function to grow a line with a relaxation step at the end
void grow_line(frame &anim, double time, int start_frame, int end_frame,
               vec &ori_1, vec &ori_2, color line_clr);
void grow_line(frame &anim, double time, vec &ori_1, vec &ori_2,
                color line_clr);


// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, int end_frame, double time, 
                  vec &ori_1, vec &ori_2, color &clr);
void animate_line(frame &anim, int start_frame, double time, 
                  vec &ori_1, vec &ori_2, color &clr);

// Function to draw layers
void draw_layers(std::vector<frame> &layer);

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, vec ori, 
                    color &clr);
void animate_circle(frame &anim, int start_frame, int end_frame, double time,
                    double radius, vec ori, color &clr);

// Function to clear a context
void clear_ctx(cairo_t* ctx);

void draw_array(frame &anim, double time, std::vector<vec> &array,
                double x_range, double y_range, color wrap_clr);
void draw_array(frame &anim, std::vector<vec> &array,
                double x_range, double y_range, color wrap_clr);

// Function to draw a bar graph of input data
void bar_graph(frame &anim, double time, int start_frame, int end_frame, 
               std::vector<int> &array, 
               double x_range, double y_range, color line_clr);
void bar_graph(frame &anim, double time, int start_frame, int end_frame, 
               std::vector<int> &array, vec ori, 
               double x_range, double y_range, color line_clr);

// Function to highlight a single bar in the bar graph
void highlight_bar(frame &anim, int start_frame, int end_frame, 
                   std::vector<int> &array, double x_range, double y_range, 
                   color highlight_clr, int element);
void highlight_bar(frame &anim, int start_frame, int end_frame, 
                   std::vector<int> &array, vec ori, 
                   double x_range, double y_range, 
                   color highlight_clr, int element);

// Function to visualize a vector of vectors of ints
void draw_permutations(frame &anim, int start_frame, int end_frame, 
                       std::vector<std::vector<int>> perms, vec ori,
                       double x_range, double y_range, color line_clr1,
                       color line_clr2);

// Function to initialize a vector of frames (layers)
std::vector<frame> init_layers(int layer_num, vec res, int fps, color bg_clr);

// Function to draw everything with standard parameters
void draw(std::vector<vec> &array);

void plot(frame &anim, std::vector<double> array, double time, int start_frame,
          int end_frame, vec &ori, vec &dim, color bar_clr, color arr_clr);
#endif
