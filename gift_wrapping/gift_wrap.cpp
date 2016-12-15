/*-------------gift_wrap.cpp--------------------------------------------------//
* 
* Purpose: To implement a simple gift-wrapping / convex hull algorithm
*
*   Notes: when scaling visualization, fiddle with parameters
*          Fix the gift_wrapping syntax
*
*-----------------------------------------------------------------------------*/

#include "gift_wrap.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initialize visualization stuff
    std::vector<frame> layers(3);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(400,300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], 0, 0, 0);

    parameter par = init(100);

/*
    for (size_t i = 0; i < par.points.size(); i++){
        std::cout << par.points[i].x << '\t' << par.points[i].y << '\n';
    }
    std::cout << par.hull[0].x << '\t' << par.hull[0].y << '\n';
*/
    gift_wrap(par, layers);

    draw_layers(layers);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to initialize random points
parameter init(int num){
    parameter par;
    par.points.reserve(num);

    // Creating random device
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        pos_dist(0,1);

    vec tmp;
    for (int i = 0; i < num; i++){
        tmp.x = pos_dist(gen);
        tmp.y = pos_dist(gen);
        par.points.push_back(tmp);
    }

    // Find the far left point 
    std::vector<vec> temp = par.points;
    std::sort(temp.begin(), temp.end(), 
              [](const vec a, const vec b){return a.x < b.x;});
    par.hull.push_back(temp[0]);
    return par;
}

// Function to wrap the points with a hull
void gift_wrap(parameter &par, std::vector<frame> &layers){

    // First, we need to draw the random distribution of points
    vec ori;

    // Arbitrarily cutting 10% of the resolution so circles can fit.
    double x_range = layers[0].res_x * 0.90;
    double y_range = layers[0].res_y * 0.90;

    // grow _circle command from ../visualization/cairo/cairo_vis.*
    for (size_t i = 0; i < par.points.size(); i++){
        ori.x = par.points[i].x*x_range + 0.05 * layers[2].res_x;
        ori.y = (1-par.points[i].y)*y_range + 0.05 * layers[2].res_y;
        grow_circle(layers[2], 1/(double)layers[2].fps,
                    ori, 5, (double)i / par.points.size());
    }

    // Resetting the frames
    layers[0].curr_frame = layers[2].curr_frame;
    layers[1].curr_frame = layers[2].curr_frame;
 
    // expand the hull
    vec curr_hull, prev_hull, next_hull;
    prev_hull.x = 0.0;
    prev_hull.y = 0.0;
    curr_hull = prev_hull;
    double threshold = 0.001, angle_tmp, final_angle;
    int i = 0;
    color line_clr = {1, 1, 1, 1};
    color wrap_clr = {0, 0, 1, 1};

    // a is curr_hull point, b is chek_point, c is new_hull line
    vec a, b, c;

    // Drawing until we return to our original point
    while (dist(curr_hull, par.hull[0]) > threshold){
        a.x = par.hull[i].x*x_range + 0.05 * layers[0].res_x;
        a.y = (1-par.hull[i].y)*y_range + 0.05 * layers[0].res_y;

        if (i == 0){
            curr_hull = par.hull[0];
        }

        final_angle = 2*M_PI;
        for (int j = 0; j < par.points.size(); j++){
            angle_tmp = angle(prev_hull, curr_hull, par.points[j]);
            b.x = par.points[j].x*x_range + 0.05 * layers[0].res_x;
            b.y = (1-par.points[j].y)*y_range + 0.05 * layers[0].res_y;

            // Drawing a single line for each frame
            // Note animate line draws over all potential frames
            cairo_move_to(layers[0].frame_ctx[layers[0].curr_frame], 
                          a.x, a.y);
            cairo_line_to(layers[0].frame_ctx[layers[0].curr_frame], 
                          b.x, b.y);
            cairo_set_source_rgba(layers[0].frame_ctx[layers[0].curr_frame],
                                  line_clr.r, line_clr.g, line_clr.b, 
                                  line_clr.a);
            cairo_stroke(layers[0].frame_ctx[layers[0].curr_frame]);

            // Increase frame number
            layers[0].curr_frame++;
            layers[1].curr_frame++;
            if (angle_tmp < final_angle){
                final_angle = angle_tmp;
                next_hull = par.points[j];
            }
        }

        // Drawing the wrapping paper or next hull
        c.x = next_hull.x*x_range + 0.05 * layers[0].res_x;
        c.y = (1-next_hull.y)*y_range + 0.05 * layers[0].res_y;
        animate_line(layers[1], layers[1].curr_frame, 0, a, c, wrap_clr);
        par.hull.push_back(next_hull);
        prev_hull = curr_hull;
        curr_hull = next_hull;
        std::cout << i << '\t' << par.hull[i].x << '\t' 
                  << par.hull[i].y << '\n';
        i++;
    }

    // Drawing outside hull
/*
    int curr_frame = layers[0].curr_frame;
    color wrap_clr = {0, 0, 1, 1};
    vec a, b;
    double draw_time = 0.1;
    for (size_t i = 0; i < par.hull.size() - 1; i++){
        a.x = par.hull[i].x*x_range + 0.05 * layers[0].res_x;
        a.y = (1-par.hull[i].y)*y_range + 0.05 * layers[0].res_y;
        b.x = par.hull[i+1].x*x_range + 0.05 * layers[0].res_x;
        b.y = (1-par.hull[i+1].y)*y_range + 0.05 * layers[0].res_y;
        animate_line(layers[0], curr_frame+i*4, 0.1, a, b, wrap_clr);
    }
*/
}

// Function to test the angle function
void test_angle(){
    // Initialize the 3 points
    vec a, b, c;
    a.x = 0;
    a.y = 0;

    b.x = 0;
    b.y = 1;

    c.x = 1;
    c.y = 1;

    double check_angle = angle(a,b,c);
    std::cout << check_angle << '\n';
}
