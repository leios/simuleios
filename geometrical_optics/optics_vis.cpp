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
    int line_width = 3;
    for (size_t i = 0; i < num_frames; ++i){
        frame_surface[i] = 
            cairo_image_surface_create(CAIRO_FORMAT_ARGB32, res_x, res_y);
        frame_ctx[i] = cairo_create(frame_surface[i]);
        cairo_set_line_cap(frame_ctx[i], CAIRO_LINE_CAP_ROUND);
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

        cairo_set_source_rgb(anim.frame_ctx[i], clr.r, clr.g, clr.b);
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
                    color clr){
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

// Function to draw lens at provided position
void draw_lens(std::vector<frame> &layer, double time, const sphere &lens){

    frame anim = layer[1];

    color lens_clr{.25,.75,1};
    animate_circle(layer[2], time * 0.5, lens.radius, lens.origin, lens_clr);

    // Now we need to fade the interior of the circle (lens) into existence
    std::vector<double> index_texture = create_index_texture(lens);
    std::vector<unsigned char> index_texture_char(index_texture.size());

    // modifying index_texture
    for (int i = 0; i < index_texture.size(); ++i){
        if (index_texture[i] > 1.0){
            index_texture[i] = 1.0;
        }
        index_texture_char[i] = index_texture[i] * 255;
    }

    // Finding number of frames available
    int draw_frames = time * 0.5 * anim.fps;
    int j = 0;
    double max_alpha;
    for (int i = anim.curr_frame + draw_frames; i < num_frames; ++i){
        if (i < anim.curr_frame + 2 * draw_frames){
            j++;
            max_alpha = (double)j * 0.5 / (double)draw_frames;
            index_plot(anim, i, index_texture_char, lens, lens_clr, max_alpha);
            //cairo_set_source_rgba(anim.frame_ctx[i], lens_clr.r, 
            //                      lens_clr.g, lens_clr.b, max_alpha);
            //cairo_arc(anim.frame_ctx[i], lens.origin.x, 
            //          lens.origin.y, lens.radius, 0, 2*M_PI);
            
            //cairo_fill(anim.frame_ctx[i]);
        }
        else{
            index_plot(anim, i, index_texture_char, lens, lens_clr, 0.5);
            //cairo_set_source_rgba(anim.frame_ctx[i], lens_clr.r, lens_clr.g, 
            //                      lens_clr.b, 0.5);
            //cairo_arc(anim.frame_ctx[i], lens.origin.x, 
            //          lens.origin.y, lens.radius, 0, 2*M_PI);
            //cairo_fill(anim.frame_ctx[i]);
        }
    }

    anim.curr_frame += draw_frames;
}

// function to create vector<double> for index_plot function
std::vector<double> create_index_texture(const sphere &lens){
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
                index_texture[k] = refractive_index_at(lens, loc);
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
void index_plot(frame &anim, int framenum,
                const sphere &lens, color lens_clr, double max_alpha){

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
                                      lens_clr.g, lens_clr.b, ior * max_alpha);
                cairo_fill(anim.frame_ctx[framenum]);
            }

        }
    }

}

// overloaded function to fill inside of lens with appropriate ior colors
void index_plot(frame &anim, int framenum, 
                std::vector<double> &index_texture, 
                const sphere &lens, color lens_clr, double max_alpha){

    // Creating the square that we will be working with.
    // We only need to store the upper left point, and then iterate through
    //     Each pixel from there.
    vec vertex;
    vertex.x = lens.origin.x - lens.radius;
    vertex.y = lens.origin.y - lens.radius;

    vec loc;
    double r_prime, ior;

    for (int i = 0; i < lens.radius * 2; ++i){
        for (int j = 0; j < lens.radius * 2; ++j){
            // Checking to see whether we are in the lens circle
            loc.x = vertex.x + i;
            loc.y = vertex.y + j;

            ior = index_texture[i * 2 * (int)lens.radius + j];
            //std::cout << "ior is: " << ior << '\n';
            cairo_rectangle(anim.frame_ctx[framenum], loc.x, loc.y, 1, 1);
            cairo_set_source_rgba(anim.frame_ctx[framenum], lens_clr.r,
                                  lens_clr.g, lens_clr.b, ior * max_alpha);
            cairo_fill(anim.frame_ctx[framenum]);
        }
    }
}

// overloaded function to fill inside of lens with appropriate ior colors
void index_plot(frame &anim, int framenum, 
                std::vector<unsigned char> &index_texture, 
                const sphere &lens, color lens_clr, double max_alpha){

    // Creating the square that we will be working with.
    // We only need to store the upper left point, and then iterate through
    //     Each pixel from there.
    vec vertex;
    vertex.x = lens.origin.x - lens.radius;
    vertex.y = lens.origin.y - lens.radius;

/*
    // Using Cairo instead, need vector<double>
    // NOTE: simple 255 factor different
    //       cairo_mask_surface
    cairo_surface_t *image = cairo_image_surface_create_for_data(
        (unsigned char *)index_texture.data(),
        CAIRO_FORMAT_A8, 2 * lens.radius, 2 * lens.radius,
        cairo_format_stride_for_width(CAIRO_FORMAT_A8, 2 * lens.radius));
    cairo_set_source_rgba(anim.frame_ctx[framenum], lens_clr.r, 
                          lens_clr.g, lens_clr.b, max_alpha);
    //cairo_set_source_surface(anim.frame_ctx[framenum], 
    //                         image,
    //                         0, 0);
    cairo_paint(anim.frame_ctx[framenum]);

    cairo_mask_surface(anim.frame_ctx[framenum], image, 0, 0);
    //                         vertex.x, vertex.y);
    //cairo_set_operator(anim.frame_ctx[framenum], CAIRO_OPERATOR_DEST_IN);
    //cairo_paint(anim.frame_ctx[framenum]);
*/
    cairo_t *cr = anim.frame_ctx[framenum];
    cairo_surface_t *image = cairo_image_surface_create_for_data(
        (unsigned char *)index_texture.data(),
        CAIRO_FORMAT_A8, 2 * lens.radius, 2 * lens.radius,
        cairo_format_stride_for_width(CAIRO_FORMAT_A8, 2 * lens.radius));

    cairo_set_source_surface(anim.frame_ctx[framenum], image,
                             vertex.x, vertex.y);
    cairo_rectangle(anim.frame_ctx[framenum], vertex.x, vertex.y, 
                    2 * lens.radius, 2 * lens.radius);
    cairo_paint(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_IN);
    cairo_set_source_rgba(cr, lens_clr.r, lens_clr.g, lens_clr.b, max_alpha);
    //cairo_paint(cr);

    cairo_fill(cr);
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

}

