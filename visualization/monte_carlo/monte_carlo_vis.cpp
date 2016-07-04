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
#include <random>

//#define num_frames 300
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

// Function to perform monte Carlo integration
void monte_carlo(frame &anim, double threshold, double box_length);

// Function to check whether point is in circle or not
bool in_circle(frame &anim, pos loc, double radius);

// Function to write area found on the lower right
void print_area(frame &anim, double area, color clr);

// Function to write the percent error on the upper left
void print_pe(frame &anim, double pe, color clr);

// Function to write the count on upper right
void print_count(frame &anim, int count);

// Function to draw a batman symbol
void draw_batman(frame &anim, double scale, pos ori);

// Function to check if we are in batman function for monte_carlo
bool is_batman(pos dot, pos ori);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    frame anim = frame(400, 300, 10, "frames/image");
    anim.init(0,0,0);

    anim.curr_frame = 1;

    //animate_square(anim, 1.0, 250, anim.origin);

    //animate_circle(anim, 1.0, 250 / 2, anim.origin);

    draw_batman(anim, anim.res_x * 0.05, anim.origin);

    //monte_carlo(anim, 0.001, 250);

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

    cairo_set_line_width(anim.frame_ctx[anim.curr_frame], 1);

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

    //anim.curr_frame += 1;

}

// Function to perform the monte carlo integration
void monte_carlo(frame &anim, double threshold, double box_length){

    // Creating random numbers with mt19937
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);

    // integer to hold the max number our vectors should count to
    int vec_count = 1, prev_print_count = 0;

    // Creating vector to hold points for visualization later
    std::vector<pos> points(1024);
    std::vector<color> pt_clrs(1024), area_clrs(1024), pe_clrs(1024);

    color pt_clr, area_clr, pe_clr;

    double count_in = 0; 

    std::vector<double> area(1024), pe_vec(1024);
    double true_area = M_PI * 0.25 * box_length * box_length;
    double temp_area;

    // distribution for box
    std::uniform_real_distribution<double> box_dist(-0.5,0.5);

    // distribution for oval?
    //std::uniform_real_distribution<double> oval_dist;

    // defining location of dot
    pos loc;

    // pe is the percent error -- setting arbitrarily high for now...
    double pe = 10;

    std::cout << "performing monte_carlo..." << '\n';

    // number of generations to wor with
    int iterations = 200;
    int final_count = 1;

    for (int i = 0; i < iterations; ++i){
        if (final_count < 1024){
            final_count *= 2;
        }
        else{
            final_count += 1024;
        }
    }

    //while (abs(pe) > threshold){
    for (int count = 1; count < final_count; ++count){

        loc.x = box_dist(gen) * box_length + anim.origin.x;
        loc.y = box_dist(gen) * box_length + anim.origin.y;

        if (in_circle(anim, loc, box_length/2)){
            count_in += 1;
            pt_clr.b = 1.0;
            pt_clr.g = 0.0;
            pt_clr.r = 0.0;
        }
        else{
            pt_clr.b = 0.0;
            pt_clr.g = 0.0;
            pt_clr.r = 1.0;
        }

        //points.push_back(loc);
        points[count - prev_print_count - 1] = loc;
        //std::cout << count - prev_print_count << '\t' 
        //          << points[count - prev_print_count - 1].x << '\t' 
        //          << points[count - prev_print_count - 1].y << '\n';
 
        //pt_clrs.push_back(pt_clr);
        pt_clrs[count - prev_print_count - 1] = pt_clr;

        temp_area = (((double)count_in/(double)count)*box_length*box_length);
        //area.push_back(temp_area);

        pe = (temp_area - true_area) / true_area;
        //pe_vec.push_back(abs(pe));

        if (abs(pe) < threshold){
            area_clr.r = 0;
            area_clr.g = 1;
            area_clr.b = 0;
            pe_clr.r = 0;
            pe_clr.g = 1;
            pe_clr.b = 0;
        }
        if (abs(pe) >= threshold){
            area_clr.r = 1;
            area_clr.g = 0;
            area_clr.b = 0;
            pe_clr.r = 1;
            pe_clr.g = 0;
            pe_clr.b = 0;

        }

        //area_clrs.push_back(area_clr);
        //pe_clrs.push_back(pe_clr);

/*
        draw_point(anim, loc, pt_clr);
        if (anim.curr_frame + 1 < num_frames){
            anim.curr_frame++;
        }
*/

        if (count - prev_print_count == vec_count){
            //std::cout << "printing..." << '\n';
            for (int j = 0; j < vec_count; ++j){
                draw_point(anim, points[j], pt_clrs[j]);
                //std::cout << points[j].x << '\t' << points[j].y << '\n';
            }
            print_area(anim, temp_area, area_clr);
            print_pe(anim, abs(pe), pe_clr);
            print_count(anim, count);
            if (vec_count < 1024){
                vec_count *= 2;
            }
            if (anim.curr_frame + 1 < num_frames){
                anim.curr_frame++;
            }
            prev_print_count = count;
        }

        //std::cout << count << '\t' << vec_count << '\t' 
        //          << prev_print_count << '\t' << temp_area 
        //          << '\t' << pe << '\n';

        std::cout << count << '\n';

    }

    std::cout << anim.curr_frame << '\n';

}

// Function to check whether point is in circle or not
bool in_circle(frame &anim, pos loc, double radius){

    double x = loc.x - anim.origin.x;
    double y = loc.y - anim.origin.y;

    if (x*x + y*y < radius * radius){
        return true;
    }
    else{
        return false;
    }
}

// Function to write area found on the lower right
void print_area(frame &anim, double area, color clr){

    // Drawing black box underneath "Area"
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], 0, 0, 0);
    cairo_rectangle(anim.frame_ctx[anim.curr_frame], 0, 0, anim.res_x, 20);
    cairo_fill(anim.frame_ctx[anim.curr_frame]);
    std::string area_txt, number;
 
    std::stringstream ss;
    ss << std::setw(3) << area;
    number = ss.str();

    area_txt = "Area: " + number;
    //std::cout << area_txt << '\n';

    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], clr.r,clr.g,clr.b);

    cairo_text_extents_t textbox;
    cairo_text_extents(anim.frame_ctx[anim.curr_frame], 
                       area_txt.c_str(), &textbox);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 20, 20);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], area_txt.c_str());

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

}

// Function to write the percent error on the upper left
void print_pe(frame &anim, double pe, color clr){

    // Drawing black box underneath "Percent Error"
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], 0, 0, 0);
    cairo_rectangle(anim.frame_ctx[anim.curr_frame], 0, 
                    anim.res_y - 20, anim.res_x, 20);
    cairo_fill(anim.frame_ctx[anim.curr_frame]);
    std::string pe_txt, number;
 
    std::stringstream ss;
    ss << std::setw(3) << pe;
    number = ss.str();

    pe_txt = "Percent Error: " + number;
    //std::cout << pe_txt << '\n';

    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], clr.r,clr.g,clr.b);

    cairo_text_extents_t textbox;
    cairo_text_extents(anim.frame_ctx[anim.curr_frame], 
                       pe_txt.c_str(), &textbox);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 20, anim.res_y);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], pe_txt.c_str());

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

}

// Function to write the count on upper right
void print_count(frame &anim, int count){
    std::string count_txt, number;
 
    std::stringstream ss;
    ss << std::setw(6) << count;
    number = ss.str();

    count_txt = "Count: " + number;

    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], 1, 1, 1);

    cairo_text_extents_t textbox;
    cairo_text_extents(anim.frame_ctx[anim.curr_frame], 
                       count_txt.c_str(), &textbox);
    cairo_move_to(anim.frame_ctx[anim.curr_frame], anim.res_x / 2, 20);
    cairo_show_text(anim.frame_ctx[anim.curr_frame], count_txt.c_str());

    cairo_stroke(anim.frame_ctx[anim.curr_frame]);

}



// Function to draw a batman symbol
void draw_batman(frame &anim, double scale, pos ori){

    int res = 1000;
    std::vector<pos> wing_l(res), wing_r(res), wing_bot(res), 
                     shoulder_l(res), shoulder_r(res), head(res);

    double pos_y, pos_x;

    // First, let's draw the batman function. It seems to be split into 6-ish

    // Creating side wings
    for (int i = 0; i < res; ++i){
        pos_y = -2.461955419944869
                +(double)i * (5.172479128660623)/(double)res;
        pos_x = 7 * sqrt(1-((pos_y * pos_y)/9.0));

        std::cout << pos_y << '\t' << pos_x << '\n';

        wing_l[i].y = ori.y - (pos_y) * scale;
        wing_l[i].x = ori.x + (pos_x) * scale;
        wing_r[i].y = ori.y - (pos_y) * scale;
        wing_r[i].x = ori.x - (pos_x) * scale;
    }


    // drawing non-ambiguous x-dependent functions (everything else)
    for (int i = 0; i < res; ++i){
        pos_x = -4.0 + (double)i * 8.0 / (double)res;
        pos_y = (abs(pos_x / 2) - 0.09137221374655434 * pos_x * pos_x - 3.0)
                + sqrt(1 - (abs(abs(pos_x)-2)-1) * (abs(abs(pos_x)-2)-1));

        std::cout << pos_x << '\t' << pos_y << '\n';
        wing_bot[i].x = ori.x + (pos_x) * scale;
        wing_bot[i].y = ori.y - (pos_y) * scale;

        pos_x = 1 + (double)i * 2 / (double)res;
        pos_y = (2.710523708715754 + (1.5 - 0.5 * pos_x)) 
                - 1.355261854357877*sqrt(4.0-(abs(pos_x)-1)*(abs(pos_x)-1)); 

        std::cout << pos_x << '\t' << pos_y << '\n';
        shoulder_l[i].x = ori.x + scale * pos_x;
        shoulder_l[i].y = ori.y - scale * pos_y;
        shoulder_r[i].x = ori.x - scale * pos_x;
        shoulder_r[i].y = ori.y - scale * pos_y;

        pos_x = -1 + (double)i * 2 / (double)res;
        if (pos_x < -0.75){
            pos_y = 9.0 + 8 * pos_x;
        }
        if (pos_x >= -0.75 && pos_x < -0.5){
            pos_y = -3 * pos_x + 0.75;
        }
        if (pos_x >= -.5 && pos_x < 0.5){
            pos_y = 2.25;
        }
        if (pos_x >= 0.5 && pos_x < 0.75){
            pos_y = 3 * pos_x + 0.75;
        }
        if (pos_x >= 0.75 && pos_x <= 1.0){
            pos_y = 9.0 - 8 * pos_x;
        }

        std::cout << pos_x << '\t' << pos_y << '\n';
        head[i].x = ori.x + scale * pos_x;
        head[i].y = ori.y - scale * pos_y;
    }

    // Draw everything

    // Drawing left wing
    cairo_move_to(anim.frame_ctx[anim.curr_frame], wing_l[0].x, wing_l[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame], 
                          wing_l[i].x - wing_l[i-1].x,
                          wing_l[i].y - wing_l[i-1].y);
        
    }
        
    // Drawing left shoulder
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                  shoulder_l[0].x, shoulder_l[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame],
                          shoulder_l[i].x - shoulder_l[i-1].x,
                          shoulder_l[i].y - shoulder_l[i-1].y);
        
    }

    // Drawing head
    cairo_move_to(anim.frame_ctx[anim.curr_frame], head[0].x, head[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame], 
                          head[i].x - head[i-1].x,
                          head[i].y - head[i-1].y);
        
    }

    // Drawing right shoulder
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                  shoulder_r[0].x, shoulder_r[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame],
                          shoulder_r[i].x - shoulder_r[i-1].x,
                          shoulder_r[i].y - shoulder_r[i-1].y);
        
    }

    //drawing right wing 
    cairo_move_to(anim.frame_ctx[anim.curr_frame], wing_r[0].x, wing_r[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame], 
                          wing_r[i].x - wing_r[i-1].x,
                          wing_r[i].y - wing_r[i-1].y);

    }


    // Drawing bottom wing
    cairo_move_to(anim.frame_ctx[anim.curr_frame], 
                  wing_bot[0].x, wing_bot[0].y);
    for (int i = 1; i < res; ++i){
        cairo_rel_line_to(anim.frame_ctx[anim.curr_frame], 
                          wing_bot[i].x - wing_bot[i-1].x,
                          wing_bot[i].y - wing_bot[i-1].y);
    }
        
    cairo_set_source_rgb(anim.frame_ctx[anim.curr_frame], 1, 1, 1);
    cairo_stroke(anim.frame_ctx[anim.curr_frame]);
}
