/*-------------optics_vis.cpp-------------------------------------------------//
*
* Purpose: To visualize a geometrical optics for LeiosOS
*
*   Notes: This will be using the cairo package, hopefully creating animations
*              I could use the subroutine-comic project, but this will be from 
*              scratch
*
*-----------------------------------------------------------------------------*/

#include "optics_vis.h"
#include "geometrical.h"

// Function to initialize the frame struct
void frame::init(){
    int line_width = 1;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        //cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
        cairo_set_line_width(frame_ctx[i], line_width);
        cairo_set_font_size(frame_ctx[i], 15.0);
    }
    bg_surface = 
        cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
    bg_ctx = cairo_create(bg_surface);
    curr_frame = 0;
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

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight){

    // Number of frames 
    int draw_frames = time * anim.fps;

    double curr_radius = 0;

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    double temp_weight;

    for (int i = anim.curr_frame; i < num_frames; ++i){
        if (i < anim.curr_frame + draw_frames){
            //expansion step
            if (i < anim.curr_frame + ceil(draw_frames * 0.5)){
                j++;
                curr_radius = (double)j * (radius * 1.25) 
                              / (double)ceil(draw_frames * 0.5);
                //std::cout << "j is: " << j << '\t' << "curr_radius is: "
                //          << curr_radius << '\n';

            }
            // Relaxation step
            else{
                k++;
                curr_radius = (radius * 1.25) + radius*((double)k * (1.0 - 1.25)
                              / (double)ceil(draw_frames * 0.5));
                //std::cout << "k is: " << k << '\t' << "curr_radius is: "
                //          << curr_radius << '\n';
            }
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      curr_radius, 0, 2*M_PI);

        }
        else{
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      radius, 0, 2*M_PI);
        }

        // Adding in a color ramp
        // Note: Ramp is arbitrarily set
        if (weight < 0.25){
            temp_weight = weight * 4.0;
            cairo_set_source_rgb(anim.frame_ctx[i], .25 + 0.75 * temp_weight, 
                                 1, .25);
        }
        else{
            temp_weight = (weight - 0.25) * 1.333333;
            cairo_set_source_rgb(anim.frame_ctx[i], 1, 
                                 1 - (0.75 * temp_weight), .25);

        }

        cairo_fill(anim.frame_ctx[i]);

        cairo_stroke(anim.frame_ctx[i]);

        
    }

    //std::cout << "finished loop" << '\n';
    anim.curr_frame += draw_frames;
    std::cout << anim.curr_frame << '\n';
}

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time,  
                  vec &ori_1, vec &ori_2, color &clr){

    // Finding number of frames
    int draw_frames = time * anim.fps;

    // internal count that definitely starts at 0;
    int j = 0;

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

        cairo_set_source_rgba(anim.frame_ctx[i], clr.r, clr.g, clr.b, clr.a);
        cairo_stroke(anim.frame_ctx[i]);

    }

    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }

}

// Function to draw all layers
void draw_layers(std::vector<frame> &layer){
    std::string pngid, number;
    for (size_t i = 0; i < num_frames; ++i){
        for (size_t j = layer.size() - 1; j > 0; --j){
            cairo_set_source_surface(layer[0].frame_ctx[i], 
                                     layer[j].frame_surface[i], 0, 0);
            cairo_paint(layer[0].frame_ctx[i]);
        }

        // Setting up number with stringstream
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        number = ss.str();

        pngid = layer[0].pngbase + number + ".png";
        //std::cout << pngid << '\n';
        cairo_surface_write_to_png(layer[0].frame_surface[i], pngid.c_str());
    }

}

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, vec ori, 
                    color &clr){
    int j = 0;

    int draw_frames = time * anim.fps;

    // drawing a white circle
    for (int i = anim.curr_frame; i < num_frames; ++i){
        j += 1;

        cairo_set_source_rgb(anim.frame_ctx[i], clr.r, clr.g, clr.b);
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
