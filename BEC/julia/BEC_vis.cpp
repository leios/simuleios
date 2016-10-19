/*-------------BEC_vis.cpp----------------------------------------------------//
*
* Purpose: to visualize the creation of a BEC in 1d
*
*-----------------------------------------------------------------------------*/

#include "BEC_vis.h"

int main(){
}

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
void draw_BEC(frame &anim, std::ofstream &file){
/*
    // Animating the drawing of the axes during that time.
    color white{1, 1, 1, 1}; 

    // Defining starting and ending position of x axis (r)
    int offset = 50;
    vec start_p = vec(0, anim.res_y - offset);
    vec end_p = vec(anim.res_x, anim.res_y - offset);
    animate_line(anim, 0, time, start_p, end_p, white);

    // Defining starting and ending position of y axis (index)
    start_p = vec(offset, anim.res_y);
    end_p = vec(offset, offset);

    animate_line(anim, 0, time, start_p, end_p, white);

    // Placing "r" on x-axis and "index" on y-axis
    
    // Defining pixels left to work with in x and y after axis are drawn
    int x_pixels = anim.res_x - 2 * offset;
    int y_pixels = anim.res_y - 2 * offset;

    anim.curr_frame += draw_frame;

    double y;

    std::string index_txt, x_txt;
    index_txt = "|Psi|^2";
    x_txt = "x";

    // Defining starting index for index sweep

    // Visualizing the refractive index as it changes with time.
    while (file << item){
        cairo_text_extents_t textbox;
        cairo_text_extents(anim.frame_ctx[i], 
                           index_txt.c_str(), &textbox);
        cairo_move_to(anim.frame_ctx[i], 20, textbox.height);
        cairo_show_text(anim.frame_ctx[i], index_txt.c_str());
        cairo_text_extents(anim.frame_ctx[i], 
                           x_txt.c_str(), &textbox); 
        cairo_move_to(anim.frame_ctx[i], anim.res_x - textbox.width,
                      anim.res_y);
        cairo_show_text(anim.frame_ctx[i], x_txt.c_str());
    
        cairo_stroke(anim.frame_ctx[anim.curr_frame]);

        for (int j = 0; j < x_pixels; ++j){
            ray_p = vec(offset + j, lens.origin.y);
            if (j == 0){
                cairo_move_to(anim.frame_ctx[i], offset, y);
            }
            else{
                cairo_line_to(anim.frame_ctx[i], j + offset, y);
                cairo_stroke(anim.frame_ctx[i]);
                cairo_move_to(anim.frame_ctx[i], j + offset, y);
            }
        }

    }
*/
}
