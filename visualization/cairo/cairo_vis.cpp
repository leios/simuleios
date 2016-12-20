/*-------------cairo_vis.cpp--------------------------------------------------//
*
* Purpose: to visualize any 2d simulation data for Simuleios
*
*-----------------------------------------------------------------------------*/

#include "cairo_vis.h"

// Function to initialize the frame struct
void frame::init(){
    int line_width = 5;
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

// Creating basic colored background
void color_bg(frame &anim, int start_layer, int r, int g, int b){
    for (int i = start_layer; i < num_frames; ++i){
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

void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight){
    grow_circle(anim, time, anim.curr_frame, num_frames, ori, radius, weight);
}

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, int start_frame, int end_frame, 
                 vec &ori, double radius, double weight){

    // Number of frames 
    int draw_frames = time * anim.fps;

    double curr_radius = 0;

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    double temp_weight;

    for (int i = start_frame; i < end_frame; ++i){
        if (i < start_frame + draw_frames){
            //expansion step
            if (i < start_frame + ceil(draw_frames * 0.5)){
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
    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }
    std::cout << anim.curr_frame << '\n';
}

void animate_line(frame &anim, int start_frame, double time,  
                  vec &ori_1, vec &ori_2, color &clr){
    animate_line(anim, start_frame, num_frames, time, ori_1, ori_2, clr);
}

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, int end_frame, double time,  
                  vec &ori_1, vec &ori_2, color &clr){

    // Finding number of frames
    int draw_frames = time * anim.fps;

    // internal count that definitely starts at 0;
    int j = 0;

    double curr_x, curr_y;

    for (int i = start_frame; i < end_frame; ++i){
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
        for (size_t j = 1; j < layer.size(); ++j){
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
    //double body_height;
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame],
                         clr.r, clr.g, clr.b);
    double radius = height * 0.33333333;
    cairo_arc(anim.frame_ctx[anim.curr_frame], pos.x, pos.y,
              radius, 0, 2*M_PI);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], pos.x, pos.y);
    cairo_set_source_rgba(anim.frame_ctx[anim.curr_frame], 0, 0, 0, 1);
    cairo_fill(anim.frame_ctx[anim.curr_frame]);
    cairo_set_source_rgba(anim.frame_ctx[anim.curr_frame], 
                          clr.r, clr.g, clr.b, clr.a);
    cairo_arc(anim.frame_ctx[anim.curr_frame], pos.x, pos.y,
              radius, 0, 2*M_PI);
    vec head_pos;
    head_pos.x = pos.x;
    head_pos.y = pos.y + radius;
    vec foot_pos;
    foot_pos.x = pos.x;
    // Check foot position y.
    foot_pos.y = pos.y + height;

    //body_height = head_pos.y - foot_pos.y;

    cairo_move_to(anim.frame_ctx[anim.curr_frame], head_pos.x, head_pos.y);
    cairo_line_to(anim.frame_ctx[anim.curr_frame], foot_pos.x, foot_pos.y);

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);
}

// Function to animate the drawing of a human
void animate_human(frame &anim, vec pos, double height, color clr, double time){
    //double body_height;
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame],
                         clr.r, clr.g, clr.b);
    double radius = height * 0.33333333;
    animate_circle(anim, time * 2 / 3, radius, pos, clr);
    vec head_pos;
    head_pos.x = pos.x;
    head_pos.y = pos.y + radius;
    vec foot_pos;
    foot_pos.x = pos.x;
    // Check foot position y.
    foot_pos.y = pos.y + height;

    //body_height = head_pos.y - foot_pos.y;

    animate_line(anim, anim.curr_frame, time / 3, head_pos, foot_pos, clr);  

}

void write_text(frame &anim, vec head_pos, vec text_pos, double head_radius,
                double font_size, std::string text){

    double offset = (double)anim.res_y * 0.02;
    // Determine size in x of box
    //double text_width = anim.res_x * 0.3333;

    // Draw line from person to text
    std::vector<double> x_line(3), y_line(3);

    cairo_move_to(anim.frame_ctx[anim.curr_frame], head_pos.x, 
                  head_pos.y - head_radius - offset);
    cairo_line_to(anim.frame_ctx[anim.curr_frame], text_pos.x, 
                  text_pos.y + offset);

    // Draw text
    cairo_set_font_size (anim.frame_ctx[anim.curr_frame], font_size);
    cairo_set_source_rgb (anim.frame_ctx[anim.curr_frame], 1.0, 1.0, 1.0);

    // Determining where to move to
    cairo_text_extents_t textbox;
    cairo_text_extents(anim.frame_ctx[anim.curr_frame], text.c_str(), &textbox);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                  text_pos.x - textbox.width / 2.0, text_pos.y);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], text.c_str());

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

}

// function to draw an array (or vector of vec's)
void draw_array(frame &anim, std::vector<vec> &array, 
                double x_range, double y_range, color wrap_clr){
    int curr_frame = anim.curr_frame;
    vec a, b;
    for (size_t i = 0; i < array.size() - 1; i++){
        a.x = array[i].x*x_range + 0.05 * anim.res_x;
        a.y = (1-array[i].y)*y_range + 0.05 * anim.res_y;
        b.x = array[i+1].x*x_range + 0.05 * anim.res_x;
        b.y = (1-array[i+1].y)*y_range + 0.05 * anim.res_y;
        animate_line(anim, curr_frame+i*4, 0.1, a, b, wrap_clr);
    }

}

// Function to clear a context -- BETA
void clear_ctx(cairo_t* ctx){
    // Set surface to translucent color (r, g, b, a)
    cairo_save (ctx);
    cairo_set_source_rgba (ctx, 0, 0, 0, 0);
    cairo_set_operator (ctx, CAIRO_OPERATOR_SOURCE);
    cairo_paint (ctx);
    cairo_restore (ctx);
}
