/*-------------huffman_vis.cpp------------------------------------------------//
*
* Purpose: To visualize a simple huffman tree for LeiosOS
*
*   Notes: This will be using the cairo package, hopefully creating animations
*          I could use the subroutine-comic project, but this will be from 
*          scratch
*
*-----------------------------------------------------------------------------*/

#include <cairo.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include "huffman.h"

#define num_frames 300
//#define num_frames 20

// Struct to hold positions
struct pos{
    double x, y;
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
    pos origin;
    std::string pngbase;

    // Function to call frame struct
    frame(int x, int y, int ps, std::string pngname);

    // Function to initialize the frame struct
    void init(int r, int g, int b);

    // Function to draw all frames in the frame struct
    void draw_frames();

    // Function to destroy all contexts and surfaces
    void destroy_all();

};

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

} 

// Function to initialize the frame struct
void frame::init(int r, int g, int b){
    int line_width = 3;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        cairo_set_source_rgb(frame_ctx[i],(double)r, (double)g, (double)b);
        //cairo_rectangle(frame_ctx[i],0,0,res_x,res_y);
        cairo_fill(frame_ctx[i]);
        cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(frame_ctx[i], line_width);
        cairo_set_font_size(frame_ctx[i], 20.0);
    }
    bg_surface = 
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
    bg_ctx = cairo_create(bg_surface);
    curr_frame = 0;
}

// Function to draw all frames in the frame struct
void frame::draw_frames(){
    std::string pngid, number;
    for (size_t i = 0; i < num_frames; ++i){
        cairo_set_source_surface(frame_ctx[i], frame_surface[i], 0, 0);
        cairo_paint(frame_ctx[i]);

        // Setting up number with stringstream
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        number = ss.str();

        pngid = pngbase + number + ".png";
        std::cout << pngid << '\n';
        cairo_surface_write_to_png(frame_surface[i], pngid.c_str());
    }

}

// Function to set the initial variables
frame::frame(int x, int y, int ps, std::string pngname){
    res_x = x;
    res_y = y;
    pngbase = pngname;
    fps = ps;
    origin.x = (double)x / 2.0;
    origin.y = (double)y / 2.0;
}

