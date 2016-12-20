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
        layers[i].create_frame(400, 300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], 0, 0, 0);

    parameter par = init(10);

/*
    for (size_t i = 0; i < par.points.size(); i++){
        std::cout << par.points[i].x << '\t' << par.points[i].y << '\n';
    }
    std::cout << par.hull[0].x << '\t' << par.hull[0].y << '\n';
*/
    //par.wrap_clr = {0, 1, 0, 1};
    //par.wrap_clr2 = {1, 1, 1, 1};

    //jarvis(par, layers);

    par.wrap_clr = {0, 0, 1, 1};
    par.wrap_clr2 = {1, 0, 0, 1};

    graham(par, layers);
    //chan(par, 4, layers);

    // Arbitrarily cutting 10% of the resolution so circles can fit.
    //double x_range = layers[0].res_x * 0.90;
    //double y_range = layers[0].res_y * 0.90;

    //grow_dist(par, layers, x_range, y_range);


    std::cout << "drawing out..." << '\n';

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

    return par;
}

// Function to draw random distribution with grow_circle command
void grow_dist(parameter &par, std::vector<frame> &layers, 
               double x_range, double y_range){

    // First, we need to draw the random distribution of points
    vec ori;

    // grow _circle command from ../visualization/cairo/cairo_vis.*
    for (size_t i = 0; i < par.points.size(); i++){
        ori.x = par.points[i].x*x_range + 0.05 * layers[2].res_x;
        ori.y = (1-par.points[i].y)*y_range + 0.05 * layers[2].res_y;
        grow_circle(layers[2], 1/(double)layers[2].fps,
                    ori, 10, (double)i / par.points.size());
    }

    // Resetting the frames
    layers[0].curr_frame = layers[2].curr_frame;
    layers[1].curr_frame = layers[2].curr_frame;
}

// Function to wrap the points with a hull
void jarvis(parameter &par, std::vector<frame> &layers){

    // Find the far left point 
    std::vector<vec> temp = par.points;
    std::sort(temp.begin(), temp.end(), 
              [](const vec a, const vec b){return a.x < b.x;});
    par.hull.push_back(temp[0]);


    // Arbitrarily cutting 10% of the resolution so circles can fit.
    double x_range = layers[0].res_x * 0.90;
    double y_range = layers[0].res_y * 0.90;

    if (!par.chan){
        grow_dist(par, layers, x_range, y_range);
    }

    // expand the hull
    vec curr_hull, prev_hull, next_hull;
    prev_hull.x = 0.0;
    prev_hull.y = 0.0;
    curr_hull = prev_hull;
    double threshold = 0.001, angle_tmp, final_angle;
    int i = 0;

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
                                  par.wrap_clr2.r, par.wrap_clr2.g, 
                                  par.wrap_clr2.b, par.wrap_clr2.a);
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
        animate_line(layers[1], layers[1].curr_frame, 0, a, c, par.wrap_clr);
        par.hull.push_back(next_hull);
        prev_hull = curr_hull;
        curr_hull = next_hull;
        std::cout << i << '\t' << par.hull[i].x << '\t' 
                  << par.hull[i].y << '\n';
        i++;
    }

    // Drawing outside hull
}

// Function to return sign of value
double sign(double value){
    double ret_val = 0;
    if (value < 0){
        ret_val = -1;
    }
    else if (value > 0){
        ret_val = 1;
    }
    return ret_val;
}

// Function to find angle if casting along origin
double cast_angle(vec v){

    double ret_angle = 0;
    if (sign(v.x) <= 0 && sign(v.y) > 0){
        ret_angle = atan(v.x/v.y) + 0.5*M_PI;
    }
    else if (sign(v.x) < 0 && sign(v.y) <= 0){
        ret_angle = atan(v.y/v.x) + M_PI;
    }
    else if (sign(v.x) >= 0 && sign(v.y) < 0){
        ret_angle = atan(-v.x/v.y) + 1.5 * M_PI;
    }
    else{
        ret_angle = atan(v.y/v.x);
    }

    return ret_angle;

}

// Finding the angle between 3 points
double angle(vec A, vec B, vec C){

/*
    vec v1;
    vec v2;
    v1.x = B.x - A.x;
    v1.y = B.y - A.y;
    v2.x = C.x - A.x;
    v2.y = C.y - A.y;

    double angle1 = cast_angle(v1);
    double angle2 = cast_angle(v2);
    double ret_angle = abs(angle1 - angle2);
    return ret_angle;
*/

    double a = dist(B,C);
    double b = dist(A,C);
    double c = dist(A,B);

    double ret_angle = acos((b*b - a*a - c*c)/(2*a*c));

    // Checking for obtuse angle
    if (sign(A.x - B.x) != sign(A.x - C.x)){
        ret_angle += 0.5*M_PI;
    }
    return ret_angle;
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

// Function to find CCW rotation
double ccw(vec a, vec b, vec c){
    return (b.x - a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}

// Function to test the ccw function
void ccw_test(){
    vec a = {0.5, 0};
    vec b = {0.5, 0.5};
    vec d = {0.75, 0.25};

    std::cout << ccw(a, b, d) << '\t' << ccw(d, b, a) << '\n';
}

// Function to wrap points in hull (GRAHAM SCAN EDITION)
void graham(parameter &par, std::vector<frame>& layers){

    // Defining the ranges and drawing distribution
    double x_range = layers[0].res_x * 0.90;
    double y_range = layers[0].res_y * 0.90;

    if (!par.chan){
        grow_dist(par, layers, x_range, y_range);
    }

    // creating hull by sorting according to polar angle from lower-most point.

    // finding the bottom-most point
    std::sort(par.points.begin(), par.points.end(), 
              [](const vec a, const vec b){return a.y < b.y;});

    // Creating a temporary point for finding the polar angle
    vec right = {1,par.points[0].y};

    std::sort(par.points.begin() + 1, par.points.end(),
              [&](const vec a, const vec b)
                {return angle(right, par.points[0], a) 
                      > angle(right, par.points[0], b);});

    par.points.push_back(par.points[0]);

    draw_array(layers[1], par.points, x_range, y_range, par.wrap_clr);

    int M = 1;
    vec temp;
    for (size_t i = 2; i < par.points.size() - 1; i++){
        while (ccw(par.points[M-1], par.points[M], par.points[i]) <= 0){
            if (M > 1){
                M--;
            }
            else if (i == par.points.size()){
                break;
            }
            else{
                i++;
            }
        }
        M++;

        //par.hull.push_back(par.points[M]);
        // Swapping the values
        temp = par.points[i];
        par.points[i] = par.points[M];
        par.points[M] = temp;

    }

    std::vector<vec>::const_iterator first = par.points.begin();
    std::vector<vec>::const_iterator last = par.points.begin() + M+1;
    par.hull = std::vector<vec>(first, last);

    par.hull.push_back(par.hull[0]);

    draw_array(layers[1], par.hull, x_range, y_range, par.wrap_clr2);
    for (size_t i = 0; i < layers.size(); ++i){
        layers[i].curr_frame = layers[1].curr_frame;
    }
    
}

// Function for Chan's algorithm
void chan(parameter &par, int subhull, std::vector<frame>& layers){

    // Defining the ranges and drawing distribution
    double x_range = layers[0].res_x * 0.90;
    double y_range = layers[0].res_y * 0.90;

    grow_dist(par, layers, x_range, y_range);
    par.chan = true;
    // finding the bottom-most point
    std::sort(par.points.begin(), par.points.end(), 
              [](const vec a, const vec b){return a.x < b.x;});

    par.wrap_clr = {0, 0, 1, 0.25};
    par.wrap_clr2 = {1, 0, 0, 0.5};

    // Split up par.points into the appropriate subhull
    std::vector<vec> graham_points;
    std::vector<vec> tmp_points = par.points;
    int res = par.points.size() / subhull;
    for (int i = 0; i < subhull; ++i){
        // first, create a subarray to work with
        std::vector<vec>::const_iterator first = tmp_points.begin() + i*res;
        std::vector<vec>::const_iterator last = tmp_points.begin() + (i+1)*res;
        par.points = std::vector<vec>(first, last);
        graham(par, layers);
        graham_points.insert(graham_points.end(), par.hull.begin(), 
                             par.hull.end());
    }

    par.wrap_clr = {0, 1, 0, 1};
    par.wrap_clr2 = {1, 1, 1, 1};

    par.points = graham_points;
    par.hull = std::vector<vec>();
    jarvis(par, layers);

}
