/*-------------cairo_vis.cpp--------------------------------------------------//
*
* Purpose: to visualize any 2d simulation data for Simuleios
*
*-----------------------------------------------------------------------------*/

#include "cairo_vis.h"

// Function to initialize the frame struct
void frame::init(){
    int line_width = 3;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        //cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(frame_ctx[i], line_width);
        cairo_set_font_size(frame_ctx[i], 50.0);
    }
    bg_surface = 
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
    bg_ctx = cairo_create(bg_surface);
    curr_frame = 0;
}

// Creating a function to destroy all contexts
void frame::destroy_all(){
    for (int i = 0; i < num_frames; i++){
        cairo_destroy(frame_ctx[i]);
        cairo_surface_destroy(frame_surface[i]);
    }
    cairo_destroy(bg_ctx);
    cairo_surface_destroy(bg_surface);
}

// Creating basic colored background
void create_bg(frame &anim, int r, int g, int b){
    for (int i = 0; i < num_frames; ++i){
        cairo_set_source_rgb(anim.frame_ctx[i],(double)r, (double)g, (double)b);
        cairo_rectangle(anim.frame_ctx[i],0,0,anim.res_x,anim.res_y);
        cairo_fill(anim.frame_ctx[i]);
    }
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
void frame::create_frame(int x, int y, int ps, std::string pngname){
    res_x = x;
    res_y = y;
    pngbase = pngname;
    fps = ps;
    origin.x = (double)x / 2.0;
    origin.y = (double)y / 2.0;
}

// Function to plot sinunoidal 
void draw_human(frame &anim, double pos, double phase, double freq, color clr){
    double x_pos;
    double y_pos;
    double body_height;

    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame],
                         clr.r, clr.g, clr.b);
    double radius = 30.0;
    cairo_arc(anim.frame_ctx[anim.curr_frame], pos, radius +10,
              radius, 0, 2*M_PI);
    vec head_pos;
    head_pos.x = pos;
    head_pos.y = 2 * radius + 10;
    vec foot_pos;
    foot_pos.x = pos;
    //foot_pos.y = head_pos.y + anim.res_y * 0.25;
    foot_pos.y = anim.res_y;

    body_height = head_pos.y - foot_pos.y;

    cairo_move_to(anim.frame_ctx[anim.curr_frame], head_pos.x, head_pos.y);
    if (freq != 0){
        for (int i = 0; i < 500; i++){
            y_pos = head_pos.y -((body_height) * 0.002 * i );
                               //* exp(-body_height * 0.002 * i));
            x_pos = pos + sin(y_pos * freq + phase) * 15;
            //x_pos = pos;
            cairo_line_to(anim.frame_ctx[anim.curr_frame], x_pos, y_pos);
            cairo_move_to(anim.frame_ctx[anim.curr_frame], x_pos, y_pos);
        }
    }
    else{
        cairo_line_to(anim.frame_ctx[anim.curr_frame], foot_pos.x, foot_pos.y);
    }

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);
}

// Fucntion to draw a human for viral visualzation
void draw_human(frame &anim, vec pos, double height, color clr){
    double body_height;

    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame],
                         clr.r, clr.g, clr.b);
    double radius = height * 0.33333333;
    cairo_arc(anim.frame_ctx[anim.curr_frame], pos.x, pos.y,
              radius, 0, 2*M_PI);
    vec head_pos;
    head_pos.x = pos.x;
    head_pos.y = pos.y - radius;
    vec foot_pos;
    foot_pos.x = pos.x;
    // Check foot position y.
    foot_pos.y = pos.y - height;

    body_height = head_pos.y - foot_pos.y;

    cairo_move_to(anim.frame_ctx[anim.curr_frame], head_pos.x, head_pos.y);
    cairo_line_to(anim.frame_ctx[anim.curr_frame], foot_pos.x, foot_pos.y);

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);
}

