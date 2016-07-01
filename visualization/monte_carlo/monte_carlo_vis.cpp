/*-------------monte_carlo_vis.cpp--------------------------------------------//
*
* Purpose: To visualize a simple monte-carlo agorithm for LeiosOS
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

#define num_frames 10

// Struct to hold positions
struct pos{
    double x, y;
    pos(int xpos, int ypos);
};


// Struct to hold all the necessary data for animations
struct frame{
    int res_x, res_y;
    int fps;
    cairo_surface_t *frame_surface[num_frames];
    cairo_t *frame_ctx[num_frames];
    pos origin = pos((double)res_x / 2.0, (double)res_y / 2.0);
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

// Function to draw an animated square
void animate_square(frame &anim, double time, double box_length, pos ori);

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, pos ori);

// Function to draw random points and color based on whether they are inside
// or outside the provided shapes, and also write the area to screen every frame
void draw_points();

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    frame anim = frame(400, 300, 10, "frames/image");
    anim.init(0,0,0);

    animate_square(anim, 1.0, anim.res_x - 10, anim.origin);

    anim.draw_frames();    
} 

// Function to initialize the frame struct
void frame::init(int r, int g, int b){
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        cairo_set_source_rgb(frame_ctx[i],(double)r, (double)g, (double)b);
        cairo_fill(frame_ctx[i]);
    }
}

// Function to draw all frames in the frame struct
void frame::draw_frames(){
    std::string pngid, number;
    for (size_t i = 0; i < num_frames; ++i){
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
}

// Function to create basic position
pos::pos(int xpos, int ypos){
    x = xpos;
    y = ypos;
}

// Function to draw an animated square
void animate_square(frame &anim, double time, double box_length, pos ori){

    // drawing a white square
    for (size_t i = 0; i < num_frames; ++i){
        cairo_set_source_rgb(anim.frame_ctx[i], 1, 1, 1);
    }
}
