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
        cairo_set_line_join(frame_ctx[i], CAIRO_LINE_JOIN_BEVEL); 
        cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
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
void create_bg(frame &anim, double r, double g, double b){
    create_bg(anim, r, g, b, 1);
}
void create_bg(frame &anim, double r, double g, double b, double a){
    for (int i = 0; i < num_frames; ++i){
        cairo_set_source_rgba(anim.frame_ctx[i], r, g, b, a);
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

void grow_circle(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, double weight){

    // Adding in a color ramp
    // Note: Ramp is arbitrarily set
    color cir_clr;
    double temp_weight;
    if (weight < 0.25){
        temp_weight = weight * 4.0;
        cir_clr = {.25 + 0.75 * temp_weight, 1, .25, 1};
    }
    else{
        temp_weight = (weight - 0.25) * 1.333333;
        cir_clr = {1, 1 - (0.75 * temp_weight), .25, 1};
    }

    grow_circle(anim, time, start_frame, end_frame, ori, radius, cir_clr);
}
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 color cir_clr){
    grow_circle(anim, time, anim.curr_frame, num_frames, ori, radius, cir_clr);
}


void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight){
    // Adding in a color ramp
    // Note: Ramp is arbitrarily set
    color cir_clr;
    double temp_weight;
    if (weight < 0.25){
        temp_weight = weight * 4.0;
        cir_clr = {.25 + 0.75 * temp_weight, 1, .25, 1};
    }
    else{
        temp_weight = (weight - 0.25) * 1.333333;
        cir_clr = {1, 1 - (0.75 * temp_weight), .25, 1};
    }

    grow_circle(anim, time, anim.curr_frame, num_frames, ori, radius, cir_clr);
}

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, int start_frame, int end_frame, 
                 vec &ori, double radius, color cir_clr){

    // Number of frames 
    int draw_frames = time * anim.fps;

    double curr_radius = 0;

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    for (int i = start_frame; i < end_frame; ++i){
        if (i < start_frame + draw_frames){
            //expansion step
            if (i < start_frame + ceil(draw_frames * 0.5)){
                j++;
                curr_radius = (double)j * (radius * 1.25) 
                              / (double)ceil(draw_frames * 0.5);

            }
            // Relaxation step
            else{
                k++;
                curr_radius = (radius * 1.25) + radius*((double)k * (1.0 - 1.25)
                              / (double)ceil(draw_frames * 0.5));
            }
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      curr_radius, 0, 2*M_PI);

        }
        else{
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, 
                      radius, 0, 2*M_PI);
        }

        cairo_set_source_rgba(anim.frame_ctx[i], cir_clr.r, cir_clr.g, 
                              cir_clr.b, cir_clr.a);

        cairo_fill(anim.frame_ctx[i]);

        cairo_stroke(anim.frame_ctx[i]);
        
    }

    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }
    std::cout << anim.curr_frame << '\n';
}

// Function to grow a square at the provided point.
void grow_square(frame &anim, double time, int start_frame, int end_frame,
                 vec &ori, double radius, color square_clr){
    vec dim = {radius, radius};
    grow_rect(anim, time, start_frame, end_frame, ori, dim, square_clr);
}
void grow_square(frame &anim, double time,
                 vec &ori, double radius, color square_clr){
    vec dim = {radius, radius};
    grow_rect(anim, time, anim.curr_frame, num_frames, ori, dim, square_clr);
}

// Functions to grow rectangles
void grow_rect(frame &anim, double time, int start_frame, int end_frame,
               vec &ori, vec &dim, color rect_clr){

    // Number of frames 
    int draw_frames = time * anim.fps;

    vec curr_dim;

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    for (int i = start_frame; i < end_frame; ++i){
        if (i < start_frame + draw_frames){
            //expansion step
            if (i < start_frame + ceil(draw_frames * 0.5)){
                j++;
                curr_dim.x = (double)j * (dim.x * 1.25) 
                             / (double)ceil(draw_frames * 0.5);
                curr_dim.y = (double)j * (dim.y * 1.25) 
                             / (double)ceil(draw_frames * 0.5);

            }
            // Relaxation step
            else{
                k++;
                curr_dim.x = (dim.x * 1.25) + dim.x*((double)k * (1.0 - 1.25)
                              / (double)ceil(draw_frames * 0.5));
                curr_dim.y = (dim.y * 1.25) + dim.y*((double)k * (1.0 - 1.25)
                              / (double)ceil(draw_frames * 0.5));
            }
            cairo_rectangle(anim.frame_ctx[i], ori.x - curr_dim.x*0.5, 
                            ori.y - curr_dim.y * 0.5, curr_dim.x,
                            curr_dim.y);

        }
        else{
            curr_dim = dim;
            cairo_rectangle(anim.frame_ctx[i], ori.x - curr_dim.x*0.5, 
                            ori.y - curr_dim.y * 0.5, curr_dim.x,
                            curr_dim.y);
        }

        cairo_set_source_rgb(anim.frame_ctx[i], 
                             rect_clr.r, rect_clr.g, rect_clr.b);
        cairo_fill(anim.frame_ctx[i]);

        cairo_stroke(anim.frame_ctx[i]);
        
    }

    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }
    std::cout << anim.curr_frame << '\n';

}
void grow_rect(frame &anim, double time, 
               vec &ori, vec &dim, color rect_clr){

    grow_rect(anim, time, anim.curr_frame, num_frames, ori, dim, rect_clr);
}


// Functions to grow a line with a relaxation step
void grow_line(frame &anim, double time, vec &ori_1, vec &ori_2,
                color line_clr){
    grow_line(anim, time, anim.curr_frame, num_frames, ori_1, ori_2, line_clr);
}


void grow_line(frame &anim, double time, int start_frame, int end_frame, 
                 vec &ori_1, vec &ori_2, color line_clr){

    // Number of frames 
    int draw_frames = time * anim.fps;

    vec curr_pos = ori_1;
    vec dist = {ori_2.x - ori_1.x, ori_2.y - ori_1.y};

    // internal counts that definitely start at 0
    int j = 0, k = 0;

    for (int i = start_frame; i < end_frame; ++i){
        if (i < start_frame + draw_frames){
            //expansion step
            if (i < start_frame + ceil(draw_frames * 0.75)){
                j++;
                curr_pos.x += (dist.x * 1.1) 
                              / (double)ceil(draw_frames * 0.75);
                curr_pos.y += (dist.y * 1.1) 
                              / (double)ceil(draw_frames * 0.75);

            }
            // Relaxation step
            else{
                k++;
                curr_pos.x = ori_1.x + (dist.x * 1.1) - dist.x*((double)k * (.1)
                              / (double)ceil(draw_frames * 0.25));
                curr_pos.y = ori_1.y + (dist.y * 1.1) - dist.y*((double)k * (.1)
                              / (double)ceil(draw_frames * 0.25));
            }
            cairo_move_to(anim.frame_ctx[i], ori_1.x, ori_1.y);
            cairo_line_to(anim.frame_ctx[i], curr_pos.x, curr_pos.y);

        }
        else{
            cairo_move_to(anim.frame_ctx[i], ori_1.x, ori_1.y);
            cairo_line_to(anim.frame_ctx[i], ori_2.x, ori_2.y);
        }

        cairo_set_source_rgb(anim.frame_ctx[i], 
                             line_clr.r, line_clr.g, line_clr.b);
        cairo_stroke(anim.frame_ctx[i]);
        
    }

    if (start_frame + draw_frames > anim.curr_frame){
        anim.curr_frame = draw_frames + start_frame;
    }
    //std::cout << anim.curr_frame << '\n';
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
void animate_circle(frame &anim, int start_frame, int end_frame, double time,
                    double radius, vec ori, color &clr){

    int j = 0;

    int draw_frames = time * anim.fps;

    // drawing a white circle
    for (int i = start_frame; i < end_frame; ++i){
        j += 1;

        cairo_set_source_rgb(anim.frame_ctx[i], clr.r, clr.g, clr.b);
        if (i <= anim.curr_frame + draw_frames){
            cairo_move_to(anim.frame_ctx[i], ori.x, ori.y - radius);
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, radius, 
                      1.5 * M_PI,(1.5 *  M_PI + (j)*2*M_PI/draw_frames));
        }
        else{
            cairo_move_to(anim.frame_ctx[i], ori.x + radius, ori.y);
            cairo_arc(anim.frame_ctx[i], ori.x, ori.y, radius, 0, 2*M_PI);
        }

        cairo_stroke(anim.frame_ctx[i]);
        
    }

    anim.curr_frame += draw_frames;

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

// Function to animate human, but stop at a particular frame
void animate_human(frame &anim, vec pos, double height, color clr, 
                   int start_frame, int end_frame, double time){
    //double body_height;
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame],
                         clr.r, clr.g, clr.b);
    double radius = height * 0.33333333;
    animate_circle(anim, start_frame, end_frame, time * 2/3, radius, pos, clr);
    vec head_pos;
    head_pos.x = pos.x;
    head_pos.y = pos.y + radius;
    vec foot_pos;
    foot_pos.x = pos.x;
    // Check foot position y.
    foot_pos.y = pos.y + height;

    //body_height = head_pos.y - foot_pos.y;

    animate_line(anim, anim.curr_frame, end_frame, time/3, 
                 head_pos, foot_pos, clr);  
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

void write_fraction(frame &anim, vec frac_pos, int num, int den,
                    double font_size, color font_color){
    // Setting initial parameters
    cairo_set_font_size(anim.frame_ctx[anim.curr_frame], font_size);
    cairo_set_source_rgba(anim.frame_ctx[anim.curr_frame], font_color.r,
                          font_color.g, font_color.b, font_color.a);

    frac_pos.x -= font_size*0.1;
    frac_pos.y -= font_size*0.15;

    // Now drawing fraction
    cairo_text_extents_t num_box, den_box, under_box;

    // Numerator first
    char num_char[256];
    sprintf(num_char,"%d",num);
    cairo_text_extents(anim.frame_ctx[anim.curr_frame],
                       num_char, &num_box);

    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                 frac_pos.x - num_box.width *0.5, frac_pos.y);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], num_char);

    // Now drawing underscore for underline
    cairo_text_extents(anim.frame_ctx[anim.curr_frame],
                       "__", &under_box);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                 frac_pos.x - under_box.width *0.4, frac_pos.y);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], "__");

    // Now for the denominator
    char den_char[256];
    sprintf(den_char,"%d",den);
    cairo_text_extents(anim.frame_ctx[anim.curr_frame],
                       den_char, &den_box);

    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                 frac_pos.x - den_box.width *0.5, frac_pos.y + font_size*1.15);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], den_char);

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);
    
}

// function to draw an array (or vector of vec's)
void draw_array(frame &anim, double time, std::vector<vec> &array, 
                double x_range, double y_range, color wrap_clr){
    int curr_frame = anim.curr_frame;
    vec a, b;
    for (size_t i = 0; i < array.size() - 1; i++){
        a.x = array[i].x*x_range + 0.5 * abs(anim.res_x - x_range);
        a.y = (1-array[i].y)*y_range - 0.5 * abs(anim.res_y - y_range);
        b.x = array[i+1].x*x_range + 0.5 * abs(anim.res_x - x_range);
        b.y = (1-array[i+1].y)*y_range - 0.5 * abs(anim.res_y - y_range);
        if (time > 0){
            animate_line(anim, curr_frame+i*4, time, a, b, wrap_clr);
        }
        else{
            animate_line(anim, curr_frame, time, a, b, wrap_clr);
        }
    }

}

void draw_array(frame &anim, std::vector<vec> &array, 
                double x_range, double y_range, color wrap_clr){

    draw_array(anim, 0.1, array, x_range, y_range, wrap_clr);
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

// Function to draw a bar graph of input data
void bar_graph(frame &anim, double time, int start_frame, int end_frame,
               std::vector<int> &array, 
               double x_range, double y_range, color line_clr){
    vec ori = {0,0};
    bar_graph(anim, time, start_frame, end_frame, array, ori, x_range,
              y_range, line_clr);
}
void bar_graph(frame &anim, double time, int start_frame, int end_frame,
               std::vector<int> &array, vec ori,
               double x_range, double y_range, color line_clr){

    // Finding the maximum element
    int max = *std::max_element(array.begin(), array.end());

    int draw_frames = time * anim.fps;

    int curr_bar = 0;
    vec bot, top;

    // bar_step is the number of timesteps in-between the drawing of each bar
    int bar_step = draw_frames / array.size(); 
    if (bar_step < 1){
        for (size_t i = 0; i < array.size(); i++){
            bot.x = i*x_range/array.size() + (anim.res_x - x_range)*0.5 + ori.x;
            bot.y = anim.res_y - (anim.res_y - y_range) * 0.5 - ori.y;

            top.x = bot.x;
            top.y = bot.y - y_range * array[i] / max;

            // Growing the line for the bar
            grow_line(anim, 0, start_frame, end_frame,
                      bot, top, line_clr);

        }
    }
    else{
        for (int i = 0; i < draw_frames; i++){
            if (i % bar_step == 0){
                // Finding the starting and ending positions
                bot.x = curr_bar*x_range/array.size() 
                        + (anim.res_x - x_range)*0.5 + ori.x;
                bot.y = anim.res_y - (anim.res_y - y_range) * 0.5 - ori.y;
    
                top.x = bot.x;
                top.y = bot.y - y_range * array[curr_bar] / max;
    
                // Growing the line for the bar
                grow_line(anim, time / array.size(), anim.curr_frame,
                          end_frame, bot, top, line_clr);
                curr_bar++;
            }
        }
    }
}

// Function to highlight a single bar in the bar graph
void highlight_bar(frame &anim, int start_frame, int end_frame,
                   std::vector<int> &array, double x_range, double y_range,
                   color highlight_clr, int element){
    vec ori = {0,0};
    highlight_bar(anim, start_frame, end_frame, array, ori, x_range, y_range,
                  highlight_clr, element);
}
void highlight_bar(frame &anim, int start_frame, int end_frame,
                   std::vector<int> &array, vec ori, 
                   double x_range, double y_range,
                   color highlight_clr, int element){

    // Finding the maximum element in our array
    int max = *std::max_element(array.begin(), array.end());

    vec top, bot;

    bot.x = element*x_range/array.size() + (anim.res_x - x_range)*0.5 + ori.x;
    bot.y = anim.res_y - (anim.res_y - y_range) * 0.5 - ori.y;

    top.x = bot.x;
    top.y = bot.y - y_range * array[element] / max;

    grow_line(anim, 0, start_frame, end_frame, bot, top, highlight_clr);


}

// Function to visualize a vector of vectors of ints
void draw_permutations(frame &anim, int start_frame, int end_frame, 
                       std::vector<std::vector<int>> perms, vec ori,
                       double x_range, double y_range, color line_clr1,
                       color line_clr2){

    color line_clr;
    //int x_index = perms.size()/2;
    //int y_index = perms.size()/2;
    int x_index = ceil(sqrt((double)perms.size()));
    int y_index = ceil(sqrt((double)perms.size()));
    double dx = x_range / x_index;
    double dy = y_range / y_index;
    vec ori_1;
    vec range = {dx, dy};
    int start = start_frame;
    int index;

    std::cout << x_index << '\t' << y_index << '\t' << dx << '\t' << dy << '\n';
    std::cout << perms.size() << '\n';;

    for (int i = 0; i < y_index; i++){
        for (int j = 0; j < x_index; j++){
            index = j + i * x_index;
            if (index < perms.size()){
                if (index % 2 == 0){
                    line_clr = line_clr2;
                }
                else{
                    line_clr = line_clr1;
                }
                std::cout << "index is: " << index << '\n';
                std::cout << "start is: " << start << '\n';
                ori_1.x = dx * (j + 0.5) + ori.x - x_range * 0.5;
                ori_1.y = y_range - (dy * (i +0.5) + ori.y + y_range * 0.5);
                //ori_1.y = y_range * 0.5;
                bar_graph(anim, 0, start, end_frame, perms[index], ori_1,
                          range.x * 0.5, range.y * 0.5, line_clr);
                if (start+1 < end_frame){
                    start++;
                }
            }
        }
    }
}

std::vector<frame> init_layers(int layer_num, vec res, int fps, color bg_clr){
    std::vector<frame> layers(layer_num);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(res.x, res.y,fps,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], bg_clr.r, bg_clr.g, bg_clr.b, bg_clr.a);

    return layers;
}

void draw(std::vector<vec> &array){
    // Initialize all visualization components
    int fps = 30;
    double res_x = 1000;
    double res_y = 1000;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 0};
    color line_clr = {1, 0, 1, 1};

    std::vector<frame> layers = init_layers(1, res, fps, bg_clr);

    // shift the array to be in range 0->1 from -1->1

    vec maximum = {1, 1};
    vec minimum = {-1, -1};
    vec temp;

    temp = *std::max_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.x < j.x;});
    maximum.x = temp.x;
    temp = *std::max_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.y < j.y;});
    maximum.y = temp.y;

    temp = *std::min_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.x < j.x;});
    minimum.x = temp.x;
    temp = *std::min_element(std::begin(array), std::end(array), 
                             [](const vec &i, const vec &j){
                                 return i.y < j.y;});
    minimum.y = temp.y;

    //std::cout << maximum.x << '\t' << maximum.y << '\n';
    //std::cout << minimum.x << '\t' << minimum.y << '\n';


    for (int i = 0; i < array.size(); i++){
        array[i].x = (array[i].x - minimum.x)/ (maximum.x - minimum.x);
        array[i].y = (array[i].y - minimum.y)/ (maximum.y - minimum.y);
    }

    draw_array(layers[0], array, res_x, res_y, line_clr);

    draw_layers(layers);

    for (int i = 0; i < layers.size(); ++i){
        layers[i].destroy_all();
    }

}
