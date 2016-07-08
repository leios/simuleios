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

//#define num_frames 300
#define num_frames 30

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

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, pos ori, double radius);

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time, 
                  pos ori_1, double radius_1, pos ori_2, double radius_2);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    frame anim = frame(400, 300, 10, "frames/image");
    anim.init(0,0,0);

    anim.curr_frame = 1;

    pos ori, ori_2, ori_3, ori_4;
    ori.x = 100;
    ori.y = 100;
    ori_2.x = 200;
    ori_2.y = 200;
    ori_3.x = 200;
    ori_3.y = 100;
    ori_4.x = 100;
    ori_4.y = 200;

    grow_circle(anim, 0.5, ori, 10);
    grow_circle(anim, 0.5, ori_2, 10);
    grow_circle(anim, 0.5, ori_3, 10);
    grow_circle(anim, 0.5, ori_4, 10);

    animate_line(anim, anim.curr_frame, 0.5, ori, 10, ori_2, 10);

    animate_line(anim, anim.curr_frame - 0.5 * anim.fps, 0.5, ori_3, 10, ori_4, 10);


    anim.draw_frames();  

} 

// Function to initialize the frame struct
void frame::init(int r, int g, int b){
    int line_width = 3;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        cairo_set_source_rgb(frame_ctx[i],(double)r, (double)g, (double)b);
        cairo_rectangle(frame_ctx[i],0,0,res_x,res_y);
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

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, pos ori, double radius){

    // Number of frames 
    int draw_frames = time * anim.fps;

    double curr_radius = 0;

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    for (int i = anim.curr_frame; i < num_frames; ++i){
        if (i < anim.curr_frame + draw_frames){
            //expansion step
            if (i < anim.curr_frame + ceil(draw_frames * 0.5)){
                j++;
                curr_radius = (double)j * (radius * 1.25) 
                              / (double)ceil(draw_frames * 0.5);
                std::cout << "j is: " << j << '\t' << "curr_radius is: "
                          << curr_radius << '\n';

            }
            // Relaxation step
            else{
                k++;
                curr_radius = (radius * 1.25) + radius*((double)k * (1.0 - 1.25)
                              / (double)ceil(draw_frames * 0.5));
                std::cout << "k is: " << k << '\t' << "curr_radius is: "
                          << curr_radius << '\n';
            }
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      curr_radius, 0, 2*M_PI);

        }
        else{
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      radius, 0, 2*M_PI);
        }

        cairo_set_source_rgb(anim.frame_ctx[i], .25, 1, .25);

        cairo_fill(anim.frame_ctx[i]);

        cairo_stroke(anim.frame_ctx[i]);

        
    }

/*
    for (int i = anim.curr_frame; i < num_frames; ++i){
        cairo_set_source_rgb(anim.frame_ctx[i], 1, 1, 1);
        j++;
        if (i < anim.curr_frame + draw_frames){
            curr_radius = (double)j * radius / (double)draw_frames;
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      curr_radius, 0, 2*M_PI);
        }
        else{
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      radius, 0, 2*M_PI);

        }

        cairo_stroke(anim.frame_ctx[i]);
        
    }
*/

    std::cout << "finished loop" << '\n';
    anim.curr_frame += draw_frames;
    std::cout << anim.curr_frame << '\n';
}

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time,  
                  pos ori_1, double radius_1, pos ori_2, double radius_2){

    // Finding number of frames
    int draw_frames = time * anim.fps;

    // internal count that definitely starts at 0;
    int j = 0;

    if (ori_1.x > ori_2.x){
        ori_1.x -= radius_1;
        ori_2.x += radius_2;
    }
    else{
        ori_1.x += radius_1;
        ori_2.x -= radius_2;
    }
    if (ori_1.y > ori_2.y){
        ori_1.y -= radius_1;
        ori_2.y += radius_2;
    }
    else{
        ori_1.y += radius_1;
        ori_2.y -= radius_2;
    }

    double curr_x, curr_y;

    for (int i = start_frame; i < num_frames; ++i){
        cairo_move_to(anim.frame_ctx[i], ori_1.x, ori_1.y);
        if (i < start_frame + draw_frames){
            j++;
            curr_x = ori_1.x + (double)j * (ori_2.x - ori_1.x)
                               / (double)draw_frames;
            curr_y = ori_1.y + (double)j * (ori_2.y - ori_1.y)
                               / (double)draw_frames;
            cairo_line_to(anim.frame_ctx[i], curr_x, curr_y);
        }
        else{
            cairo_line_to(anim.frame_ctx[i], ori_2.x, ori_2.y);
        }

        cairo_set_source_rgb(anim.frame_ctx[i], 1, 1, 1);
        cairo_stroke(anim.frame_ctx[i]);

/*
        // redraw circles
        cairo_set_source_rgb(anim.frame_ctx[i], 0.25, 1, 0.25);
        cairo_arc(anim.frame_ctx[i], ori_1.x, ori_1.y, radius_1, 0, 2*M_PI);
        cairo_arc(anim.frame_ctx[i], ori_2.x, ori_2.y, radius_2, 0, 2*M_PI);

        cairo_fill(anim.frame_ctx[i]);
*/

        //cairo_stroke(anim.frame_ctx[i]);

    }

    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }

}

