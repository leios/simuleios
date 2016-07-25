/*-------------huffman_vis.cpp------------------------------------------------//
*
* Purpose: To visualize a simple huffman tree for LeiosOS
*
*   Notes: This will be using the cairo package, hopefully creating animations
*              I could use the subroutine-comic project, but this will be from 
*              scratch
*          In draw_tree, nodes are being drawn backwards
*          find a way to place weight number on each node
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
#define num_frames 300

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
void grow_circle(frame &anim, double time, pos &ori, double radius, 
                 double weight);

// Function to animate a line from two points
void animate_line(frame &anim, int start_frame, double time, 
                  pos &ori_1, pos &ori_2, color &clr);

// Function to draw huffman tree
void draw_external(frame &anim, double time, huffman_tree &tree);

// Function to draw internal_nodes
void draw_internal(frame &anim, double time, node_queue &regenerated_nodes, 
                   huffman_tree &final_tree);

// Overloaded create_nodes function for root node
// This time we are recreating the tree for later
node_queue regenerate_nodes(frame &anim, node *root, 
                            std::unordered_map<char, std::string> &bitmap, 
                            int alphabet_size);

// Function to draw layers
void draw_layers(std::vector<frame> &layer);

// Function to get x and y positions of external nodes:
void draw_tree(frame &anim, int &count_x, node* root, int level, 
               node_queue &regenerated_nodes, int alphabet_size, int max_level);

// Function to visualize the encoding process
void draw_encoding(frame &anim, std::unordered_map<char, std::string> &bitmap,
                   node *root);

// Function to drop text under leaf node
void drop_text(frame &anim, std::string &codeword, double weight, pos &ori);

// Function to place the weights
void draw_weights(frame &anim, double weight, pos &ori);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    std::vector<frame> layer(3);
    for (size_t i = 0; i < layer.size(); ++i){
        layer[i].create_frame(600, 450, 10, "frames/image");
        layer[i].init();

        layer[i].curr_frame = 1;
    }

    create_bg(layer[0], 0, 0, 0);

    // encoding with 2-pass huffman
    std::string phrase = "Jack and Jill went up the hill to fetch a pail of water. Jack fell down and broke his crown and Jill came Tumbling after!";
    huffman_tree final_tree = two_pass_huffman(phrase);
    decode(final_tree);

    std::cout << "final_tree root weight is: " 
              << final_tree.root->weight << '\n';

    node_queue regenerated_nodes = regenerate_nodes(layer[1],final_tree.root,
                                                    final_tree.bitmap,
                                                    final_tree.alphabet_size);

/*
    // Destrying the regenerated_nodes and printing them out.
    node *temp_node;
    while (regenerated_nodes.size() > 0){
        temp_node = regenerated_nodes.top();
        regenerated_nodes.pop();
        std::cout << temp_node->ori.x << '\t' << temp_node->ori.y << '\t' 
                  << temp_node-> weight << '\n'; 
    }
*/

    layer[2].curr_frame = layer[1].curr_frame;
    draw_internal(layer[2], 10.0, regenerated_nodes, final_tree);

    draw_encoding(layer[2], final_tree.bitmap, final_tree.root);

    draw_layers(layer);

} 

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
void grow_circle(frame &anim, double time, pos &ori, double radius, 
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
                  pos &ori_1, pos &ori_2, color &clr){

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

// Function to draw internal_nodes
void draw_internal(frame &anim, double time, node_queue &regenerated_nodes, 
                   huffman_tree &final_tree){

    color line_clr;
    line_clr.r = 1; line_clr.b = 1; line_clr.g = 1;
    int num_lines = 40;

    node *temp_node;

    // Going through all regenerated nodes and drawing the lines to the
    // internal nodes. Ignoring external nodes, which have been drawn in 
    // the draw_tree function.
    while (regenerated_nodes.size() > 0){
        temp_node = regenerated_nodes.top();
        regenerated_nodes.pop();

        // Are we on an internal node?
        if (!temp_node->key){
            // Left line
            animate_line(anim,anim.curr_frame,time/num_lines,
                         temp_node->left->ori,temp_node->ori, line_clr);
            animate_line(anim,anim.curr_frame-((time/num_lines) * anim.fps)+1,
                         time/num_lines,temp_node->right->ori,temp_node->ori,
                          line_clr);
        }
    }

}

// Similar to create_nodes... This time we are recreating the tree for later
node_queue regenerate_nodes(frame &anim, node *root, 
                            std::unordered_map<char, std::string> &bitmap,
                            int alphabet_size){
    node_queue regenerated_nodes;

    // Performs Depth-first search and push back the priorirty queue
    int count_x = 0;
    draw_tree(anim, count_x, root, 1, regenerated_nodes, 
              alphabet_size, 8);
    
    return regenerated_nodes;
}

void draw_tree(frame &anim, int &count_x, node* root, int level, 
               node_queue &regenerated_nodes, int alphabet_size, int max_level){
    root->ori.y = ((level - 1) * (anim.res_y - max_level * 10 )/max_level) + 5;
    //root->ori.y = (1-(root->weight / 127.0)) * anim.res_y;
    regenerated_nodes.push(root);

    if (root->right){
        draw_tree(anim, count_x, root->right, level + 1, regenerated_nodes,
                  alphabet_size,max_level);
    }
    if (root->left){
        draw_tree(anim, count_x, root->left, level + 1, regenerated_nodes,
                  alphabet_size, max_level);
    }

    if (!root->left && !root->right){
        root->ori.x = ((((double)count_x+0.5)/(double)alphabet_size)*anim.res_x)
                      * 0.85 + anim.res_x * 0.0725;
        count_x += 1;
        //std::cout << "weight is: " << root->weight << '\n';
        grow_circle(anim, 0.25, 
                    root->ori, 10 + (root->weight * 0.5), root->weight/24.0);

        char test[] = { root->key, '\0' };

        // Placing text in circle
        for (int j = anim.curr_frame; j < num_frames; ++j){
            cairo_set_source_rgb(anim.frame_ctx[j], 0, 0, 0);
            cairo_text_extents_t textbox;
            cairo_text_extents(anim.frame_ctx[j], 
                               test,
                               &textbox);
            cairo_move_to(anim.frame_ctx[j], 
                          root->ori.x - textbox.width / 2.0,
                          root->ori.y + textbox.height / 2.0);
            cairo_show_text(anim.frame_ctx[j], test);
            cairo_stroke(anim.frame_ctx[j]);
        }

        //draw_weights(anim, root->weight, root->ori);

    }
    else{
        root->ori.x = (root->left->ori.x + root->right->ori.x) * 0.5;
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
        std::cout << pngid << '\n';
        cairo_surface_write_to_png(layer[0].frame_surface[i], pngid.c_str());
    }

}

// Function to draw encoding scheme
void draw_encoding(frame &anim, std::unordered_map<char, std::string> &bitmap,
                   node *root){

    // Color of visual encoding line
    color clr_encoding{ 0.3, 0.3, 1.0};

    // Continually draw blue lines to leaf node
    if (root->right){
        animate_line(anim, anim.curr_frame, 0.10, root->ori,  
                     root->right->ori, clr_encoding);

        draw_encoding(anim, bitmap, root->right);
    }
    if (root->left){

        animate_line(anim, anim.curr_frame, 0.10, root->ori,  
                     root->left->ori, clr_encoding);

        draw_encoding(anim, bitmap, root->left);
    }

    // If on leaf node, drop encoding string
    if (!root->right && !root->left){
        drop_text(anim, bitmap[root->key], root->weight, root->ori);
    }
}

// Function to drop text under leaf node
void drop_text(frame &anim, std::string &codeword, double weight, pos &ori){

    // Going through each letter in codeword
    for (size_t i = 0; i < codeword.size(); ++i){
        char test[] = { codeword[i], '\0' };

        // Going through each frame in layer
        for (int j = anim.curr_frame; j < num_frames; ++j){

            cairo_set_source_rgb(anim.frame_ctx[j], 1, 1, 1);
            cairo_text_extents_t textbox;
            cairo_text_extents(anim.frame_ctx[j], test,
                               &textbox);

            // y height is location - text height - fontsize * i - size of leaf
            //    (plus arbitrary 2 pixel offset)
            cairo_move_to(anim.frame_ctx[j], 
                          ori.x - textbox.width / 2.0,
                          ori.y + textbox.height + i * 15 
                          + 12 + (weight * 0.5));
            cairo_show_text(anim.frame_ctx[j], test);
            cairo_stroke(anim.frame_ctx[j]);
        }
        anim.curr_frame +=1;
    }

}

// Function to place the weights
void draw_weights(frame &anim, double weight, pos &ori){
    std::string weighttext = std::to_string((int)weight);
    std::cout << weighttext <<'\n';
    for (size_t i = anim.curr_frame; i < num_frames; ++i){
        cairo_set_source_rgb(anim.frame_ctx[i], 1, 1, 1);
        cairo_text_extents_t textbox;
        cairo_text_extents(anim.frame_ctx[i], weighttext.c_str(), 
                           &textbox);
        cairo_move_to(anim.frame_ctx[i],
                      ori.x - textbox.width / 2,
                      ori.y + textbox.height + 12 + weight * 0.5);
        cairo_show_text(anim.frame_ctx[i], weighttext.c_str());


/*
        cairo_text_extents_t linebox;
        cairo_text_extents(anim.frame_ctx[i], "-", &linebox);
        cairo_move_to(anim.frame_ctx[i],
                      ori.x - linebox.width / 2,
                      ori.y + linebox.height + 12 + weight * 0.5 + 15);
        cairo_show_text(anim.frame_ctx[i], "-");

        cairo_stroke(anim.frame_ctx[i]);
*/
    }
}
