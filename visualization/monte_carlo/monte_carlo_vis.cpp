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

#define num_frames 20

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

// Function to draw an animated square
void animate_square(frame &anim, double time, double box_length, pos ori);

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, pos ori);

// Function to draw random points and color based on whether they are inside
// or outside the provided shapes, and also write the area to screen every frame
void draw_point(frame &anim, pos ori, color clr);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    frame anim = frame(400, 300, 10, "frames/image");
    anim.init(0,0,0);

    anim.curr_frame = 1;

    animate_square(anim, 1.0, 140, anim.origin);

    animate_circle(anim, 1.0, 140 / 2, anim.origin);

    color clr;
    clr.r = 1; clr.b = 0; clr.g = 0;
    draw_point(anim, anim.origin, clr);

    anim.draw_frames();    
} 

// Function to initialize the frame struct
void frame::init(int r, int g, int b){
    int line_width = 5;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        cairo_set_source_rgb(frame_ctx[i],(double)r, (double)g, (double)b);
        //cairo_rectangle(frame_ctx[i],0,0,res_x,res_y);
        cairo_fill(frame_ctx[i]);
        cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(frame_ctx[i], line_width);
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

// Function to create basic position

// Function to draw an animated square
void animate_square(frame &anim, double time, double box_length, pos ori){

    int j = 0;

    // creating number of frames to work with divisible by 4
    int tot_frames = time * anim.fps;
    int draw_frames = tot_frames - tot_frames % 4;
    double x_pos, y_pos;

    int count_up = draw_frames / 4;

    // drawing a white square
    for (int i = anim.curr_frame; i < num_frames; ++i){
        j += 1;

        x_pos = ori.x - 0.5 * box_length;
        y_pos = ori.y - 0.5 * box_length;

        if (i < anim.curr_frame + draw_frames / 4){
            cairo_move_to(anim.frame_ctx[i], x_pos, y_pos);
            cairo_line_to(anim.frame_ctx[i], 
                          x_pos + box_length * ((double)j) / count_up,
                          y_pos);
        }
        if (i < anim.curr_frame + draw_frames / 2 && 
            i >= anim.curr_frame + draw_frames / 4){
            // Draw initial upper line
            cairo_move_to(anim.frame_ctx[i], x_pos, y_pos);
            cairo_rel_line_to(anim.frame_ctx[i], box_length, 0);
            cairo_rel_line_to(anim.frame_ctx[i], 0, 
                              box_length*((((double)j)/count_up)-1));
        }
        if (i < anim.curr_frame + 0.75 * draw_frames && 
            i >= anim.curr_frame + draw_frames / 2){
            // Draw initial upper line
            cairo_move_to(anim.frame_ctx[i], x_pos, y_pos);
            cairo_rel_line_to(anim.frame_ctx[i], box_length, 0);

            // Draw line on right
            cairo_rel_line_to(anim.frame_ctx[i], 0, box_length);
            cairo_rel_line_to(anim.frame_ctx[i],
                              -box_length*((((double)j)/count_up)-2.0),0);

        }
        if (i >= anim.curr_frame + draw_frames * 0.75  && 
            i < anim.curr_frame + draw_frames){
            // Draw initial upper line
            cairo_move_to(anim.frame_ctx[i], x_pos, y_pos);

            // Draw initial upper line
            cairo_rel_line_to(anim.frame_ctx[i], box_length, 0);

            // Draw line on right
            cairo_rel_line_to(anim.frame_ctx[i], 0, box_length);

            // Draw bottom line
            cairo_rel_line_to(anim.frame_ctx[i],-box_length,0);

            cairo_rel_line_to(anim.frame_ctx[i], 0, 
                              - box_length*((((double)j)/count_up)-3.0));

        }

        if (i >= anim.curr_frame + draw_frames){
            // Draw initial upper line
            cairo_move_to(anim.frame_ctx[i], x_pos, y_pos);

            // Draw initial upper line
            cairo_rel_line_to(anim.frame_ctx[i], box_length, 0);

            // Draw line on right
            cairo_rel_line_to(anim.frame_ctx[i], 0, box_length);

            // Draw bottom line
            cairo_rel_line_to(anim.frame_ctx[i],-box_length,0);

            cairo_rel_line_to(anim.frame_ctx[i], 0, -box_length);
        }

        cairo_set_source_rgb(anim.frame_ctx[i], 1, 1, 1);
        cairo_set_line_join(anim.frame_ctx[i], CAIRO_LINE_JOIN_ROUND);
        cairo_stroke(anim.frame_ctx[i]);
        
        
    }

    anim.curr_frame += draw_frames;

    std::cout << anim.curr_frame << '\n';
    anim.bg_ctx = anim.frame_ctx[anim.curr_frame];
}

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, pos ori){
    int j = 0;

    int draw_frames = time * anim.fps;

    // drawing a white circle
    for (int i = anim.curr_frame; i < num_frames; ++i){
        j += 1;

        if (i <= anim.curr_frame + draw_frames){
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, radius, 
                      1.5 * M_PI,(1.5 *  M_PI + (j)*2*M_PI/draw_frames));
        }
        else{
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, radius, 0, 2*M_PI);
        }

        cairo_stroke(anim.frame_ctx[i]);
        
    }

    anim.curr_frame += draw_frames;

}

// Function to draw point, to be used with monte carlo
void draw_point(frame &anim, pos ori, color clr){
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], clr.r, clr.b, clr.g);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], ori.x, ori.y);
    cairo_set_line_cap(anim.frame_ctx[anim.curr_frame], CAIRO_LINE_CAP_ROUND);
    cairo_line_to(anim.frame_ctx[anim.curr_frame], ori.x, ori.y);

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

    anim.curr_frame += 1;

}
