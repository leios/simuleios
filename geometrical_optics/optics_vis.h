/*------------optics_vis.h----------------------------------------------------//
*
* Purpose: Header file for optics_vis.cpp, holds all functions and structures
*
*   Notes: This will be using Cairo, be careful
*
*-----------------------------------------------------------------------------*/

#ifndef OPTICS_VIS_H
#define OPTICS_VIS_H

#include <cairo.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
#include <sstream>
#include <vector>

//#define num_frames 300
#define num_frames 300

template<typename> struct sphere;

// A very simple vector type with operators that are used in this file
struct vec {
    double x, y;

    vec() : x(0.0), y(0.0) {}
    vec(double x0, double y0) : x(x0), y(y0) {}
};


// Struct for colors
struct color{
    double r, g, b, a;
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
    vec origin;
    std::string pngbase;

    // Function to call frame struct
    void create_frame(int x, int y, int ps, std::string pngname);

    // Function to initialize the frame struct
    void init();

    // Function to draw all frames in the frame struct
    void draw_frames();

    // Function to destroy all contexts and surfaces
    void destroy_all();

};

// Function to create basic colored background
void create_bg(frame &anim, int r, int g, int b);

// Function to grow a circle at a provided point
void grow_circle(frame &anim, double time, vec &ori, double radius, 
                 double weight);

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time, 
                  vec &ori_1, vec &ori_2, color &clr);

// Function to draw layers
void draw_layers(std::vector<frame> &layer);

// Function to draw an animated circle
void animate_circle(frame &anim, double time, double radius, vec ori, 
                    color &clr);

// Function to draw lens at provided position
template <typename T>
void draw_lens(std::vector<frame> &layer, double time, const sphere<T> &lens);

// function to create vector<double> for index_plot function
template <typename T>
std::vector<double> create_index_texture(const sphere<T> &lens);

// function to fill inside of lens with appropriate refractive index colors
template <typename T>
void index_plot(frame &anim, int framenum,
                const sphere<T> &lens, color &lens_clr);

// overloaded function to fill inside of lens with appropriate ior colors
template <typename T>
void index_plot(frame &anim, int framenum, 
                std::vector<double> &index_texture, 
                const sphere<T> &lens, color &lens_clr);

// overloaded function to fill inside of lens with appropriate ior colors
template <typename T>
void index_plot(frame &anim, int framenum, 
                cairo_surface_t *image, 
                const sphere<T> &lens, color &lens_clr);

/*----------------------------------------------------------------------------//
* TEMPLATES
*-----------------------------------------------------------------------------*/

// Function to draw lens at provided position
template <typename T>
void draw_lens(std::vector<frame> &layer, double time, const sphere<T> &lens){

    frame anim = layer[1];

    color lens_clr{.25,.75,1, 1};
    animate_circle(layer[2], time * 0.5, lens.radius, lens.origin, lens_clr);

    // Now we need to fade the interior of the circle (lens) into existence
    std::vector<double> index_texture = create_index_texture(lens);
    std::vector<unsigned char> index_texture_char(index_texture.size());

    // modifying index_texture
    for (size_t i = 0; i < index_texture.size(); ++i){
        if (index_texture[i] > 1.0){
            index_texture[i] = 1.0;
        }
        index_texture_char[i] = index_texture[i] * 255;
    }

    cairo_surface_t *image = cairo_image_surface_create_for_data(
        (unsigned char *)index_texture_char.data(),
        CAIRO_FORMAT_A8, 2 * lens.radius, 2 * lens.radius,
        cairo_format_stride_for_width(CAIRO_FORMAT_A8, 2 * lens.radius));


    // Finding number of frames available
    int draw_frames = time * 0.5 * anim.fps;
    int j = 0;
    for (int i = anim.curr_frame + draw_frames; i < num_frames; ++i){
        if (i < anim.curr_frame + 2 * draw_frames){
            j++;
            lens_clr.a = (double)j / (double)draw_frames;
            index_plot(anim, i, image, lens, lens_clr);
        }
        else{
            index_plot(anim, i, image, lens, lens_clr);
        }
    }

    for (size_t i = 0; i < layer.size(); ++i){
        layer[i].curr_frame += time * layer[i].fps;
    }

}

// Function for drawing lens for propagate_mod function
template <typename T>
void draw_lens_for_frame(frame &anim, const sphere<T> &lens){

    color lens_clr{.25,.75,1, 1};

    // Now we need to fade the interior of the circle (lens) into existence
    std::vector<double> index_texture = create_index_texture(lens);
    std::vector<unsigned char> index_texture_char(index_texture.size());

    // modifying index_texture
    for (size_t i = 0; i < index_texture.size(); ++i){
        if (index_texture[i] > 1.0){
            index_texture[i] = 1.0;
        }
        index_texture_char[i] = index_texture[i] * 255;
    }

    cairo_surface_t *image = cairo_image_surface_create_for_data(
        (unsigned char *)index_texture_char.data(),
        CAIRO_FORMAT_A8, 2 * lens.radius, 2 * lens.radius,
        cairo_format_stride_for_width(CAIRO_FORMAT_A8, 2 * lens.radius));


    // Finding number of frames available
    index_plot(anim, anim.curr_frame, image, lens, lens_clr);

}

// function to create vector<double> for index_plot function
template <typename T>
std::vector<double> create_index_texture(const sphere<T> &lens){
    int vec_size = 4 * (int)lens.radius * (int)lens.radius, k;
    std::vector<double> index_texture(vec_size);
    std::vector<double> index_texture_char(vec_size);
    double max_index = 0, dist;
    vec loc;
    for (int i = 0; i < (2 * lens.radius); ++i){
        for (int j = 0; j < (2 * lens.radius); ++j){
            // Checking to see if we are in the circle or not.

            k = i * 2 *(int)lens.radius + j;
            loc.x = i + lens.origin.x - lens.radius;
            loc.y = j + lens.origin.y - lens.radius;

            dist = distance(lens.origin, loc);

            // In circle
            if (dist < lens.radius){
                index_texture[k] = lens.refractive_index_at(loc);
                if (index_texture[k] > max_index){
                    max_index = index_texture[k];
                }
            }
            // Outside of circle
            else{
                index_texture[k] = 0.0;
            }
        }
    }

    std::cout << "max index is: " << max_index << '\n';

    // Normalizing index_texture
    for (int i = 0; i < 2 * lens.radius; ++i){
        for (int j = 0; j < 2 * lens.radius; ++j){
            k = i * 2 *(int)lens.radius + j;
            //index_texture[k] /= max_index;
            index_texture[k] *= 0.2;
/*
            if (index_texture[k] > 1.0){
                index_texture[k] = 1.0;
            }
            index_texture_char[k] = index_texture[k] * 255;
*/
        }
    }


    // NOTE: return index_texture_char if using cairo
    return index_texture;
}

// function to fill inside of lens with appropriate refractive index colors
// Note: Index plot should read in only a square of values to plot 
//       (vector<double> will be created in a separate function
template <typename T>
void index_plot(frame &anim, int framenum,
                const sphere<T> &lens, color &lens_clr){

    // Creating the square that we will be working with.
    // We only need to store the upper left point, and then iterate through
    //     Each pixel from there.
    vec vertex;
    vertex.x = lens.origin.x - lens.radius;
    vertex.y = lens.origin.y - lens.radius;

/*
    std::cout << "vertices are: " << '\n';
    std::cout << vertex.x << '\t' << vertex.y << '\n';

    std::cout << "circle originx, originy, and radius are: "<< '\n';
    std::cout << lens.origin.x << '\t' << lens.origin.y << '\t' << lens.radius
              << '\n';
*/

    vec loc;
    double r_prime, ior;

    for (int i = vertex.x; i < vertex.x + lens.radius * 2; ++i){
        for (int j = vertex.y; j < vertex.y + lens.radius * 2; ++j){
            // Checking to see whether we are in the lens circle
            loc.x = i;
            loc.y = j;

            r_prime = distance(loc, lens.origin);

            if (r_prime < lens.radius){
                ior = refractive_index_at(lens, loc);
                cairo_rectangle(anim.frame_ctx[framenum], loc.x, loc.y, 1, 1);
                cairo_set_source_rgba(anim.frame_ctx[framenum], lens_clr.r,
                                      lens_clr.g, lens_clr.b, ior * lens_clr.a);
                cairo_fill(anim.frame_ctx[framenum]);
            }

        }
    }

}

// overloaded function to fill inside of lens with appropriate ior colors
template <typename T>
void index_plot(frame &anim, int framenum, 
                std::vector<double> &index_texture, 
                const sphere<T> &lens, color &lens_clr){

    // Creating the square that we will be working with.
    // We only need to store the upper left point, and then iterate through
    //     Each pixel from there.
    vec vertex;
    vertex.x = lens.origin.x - lens.radius;
    vertex.y = lens.origin.y - lens.radius;

    vec loc;
    double ior;

    for (int i = 0; i < lens.radius * 2; ++i){
        for (int j = 0; j < lens.radius * 2; ++j){
            // Checking to see whether we are in the lens circle
            loc.x = vertex.x + i;
            loc.y = vertex.y + j;

            ior = index_texture[i * 2 * (int)lens.radius + j];
            //std::cout << "ior is: " << ior << '\n';
            cairo_rectangle(anim.frame_ctx[framenum], loc.x, loc.y, 1, 1);
            cairo_set_source_rgba(anim.frame_ctx[framenum], lens_clr.r,
                                  lens_clr.g, lens_clr.b, ior * lens_clr.a);
            cairo_fill(anim.frame_ctx[framenum]);
        }
    }
}

// overloaded function to fill inside of lens with appropriate ior colors
template <typename T>
void index_plot(frame &anim, int framenum, 
                cairo_surface_t *image, 
                const sphere<T> &lens, color &lens_clr){

    // Creating the square that we will be working with.
    // We only need to store the upper left point, and then iterate through
    //     Each pixel from there.
    vec vertex;
    vertex.x = lens.origin.x - lens.radius;
    vertex.y = lens.origin.y - lens.radius;

    cairo_t *cr = anim.frame_ctx[framenum];

    cairo_set_source_surface(anim.frame_ctx[framenum], image,
                             vertex.x, vertex.y);
    cairo_rectangle(anim.frame_ctx[framenum], vertex.x, vertex.y, 
                    2 * lens.radius, 2 * lens.radius);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_IN);
    cairo_set_source_rgba(cr, lens_clr.r, lens_clr.g, lens_clr.b, lens_clr.a);
    //cairo_paint(cr);

    cairo_fill(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

}


#endif 
