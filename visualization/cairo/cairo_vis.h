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

// Creating basic colored background
void color_bg(frame &anim, int start_layer, int r, int g, int b);

// Function to draw human stuff
void draw_human(frame &anim, double pos, double phase, double freq, color clr);
void draw_human(frame &anim, vec pos, double height, color clr);
void animate_human(frame &anim, vec pos, double height, color clr, double time);
void write_text(frame &anim, vec head_pos, vec text_pos, double head_radius,
                double font_size, std::string text);

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, double weight);
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight);

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

// Function to clear a context
void clear_ctx(cairo_t* ctx);

void draw_array(frame &anim, double time, std::vector<vec> &array,
                double x_range, double y_range, color wrap_clr);
void draw_array(frame &anim, std::vector<vec> &array,
                double x_range, double y_range, color wrap_clr);

// Function to draw a bar graph of input data
void bar_graph(frame &anim, double time, std::vector<int> &array, 
               double x_range, double y_range, color line_clr);

// Function to highlight a single bar in the bar graph
void highlight_bar(frame &anim, std::vector<int> &array, 
                   double x_range, double y_range, color highlight_clr, 
                   int element);

#endif
