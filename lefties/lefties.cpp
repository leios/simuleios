/*------------gauss_vis.cpp----------------------------------------------------/
*
* Purpose: Simple visualization script for difference scenes for computus video
*          2020
*
*   Notes: Compilation instructions:
*              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o main lefties.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
*
*-----------------------------------------------------------------------------*/
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

#include "include/camera.h"
#include "include/scene.h"

std::vector<double> normalize(std::vector<double> points){
    double max_value = *std::max_element(points.begin(), points.end());

    for (int i = 0; i < points.size(); ++i){
        points[i] /= max_value;
    }

    return points;
}

std::vector<vec> normalize(std::vector<vec> points){
    double max_value = 0.0;
    for (int i = 0; i < points.size(); ++i){
         if (points[i].y > max_value){
             max_value = points[i].y;
         }
    }

    for (int i = 0; i < points.size(); ++i){
        points[i].y /= max_value;
    }

    return points;
}

std::vector<double> GM_dist(int res, vec dim, double beta){
    std::vector<double> GM_points(res);
    for (int i = 0; i < res; ++i) {
        //GM_points.emplace_back(i * dim.x/res, std::exp(i * beta));
        GM_points[i] = std::exp(i * beta);
    }

    return GM_points;
}

// TODO: - fix bug in which the frames necessary to draw line are equal to
//         or less than number of points in the line
//       - add tics
//       - allow for different colored plots
//       - axis labels
void plot(std::vector<double> points, std::string title, vec res, vec origin,
          int start_frame, int end_frame, scene& world){

    // Finding frame information
    int axis_end_frame = start_frame + (end_frame-start_frame)/4;
    int curve_end_frame = axis_end_frame + points.size();

    // Setting title to be centered on the plot...
    // TODO: Center Title
    vec title_loc = {res.x/2, origin.y-res.y};
    auto title_box = std::make_shared<text>(title_loc, 50, title);

    // Creating y axis
    vec y_start = vec{origin.x, origin.y};
    vec y_end = vec{origin.x, origin.y-res.y};
    auto y_axis = std::make_shared<line>(y_start, y_start);

    // Creating x-axis
    vec x_start = vec{origin.x, origin.y};
    vec x_end = vec{origin.x + res.x, origin.y};
    auto x_axis = std::make_shared<line>(x_start, x_start);

    // Create content for plot
    std::vector<vec> vec_points(points.size());
    std::vector<double> tmp_points = points;
    points = normalize(points);
    for (int i = 0; i < points.size(); ++i){ 
        vec_points[i].x = i * res.x/vec_points.size();
        vec_points[i].y = points[i]*res.y;
    }

    auto content = std::make_shared<curve>(std::vector<vec>(), origin);
    //auto content = std::make_shared<curve>(vec_points, origin);

    // Animating all content
    y_axis->add_animator<vec_animator>(start_frame,
                                       axis_end_frame,
                                       &y_axis->end,
                                       y_start, y_end);

    x_axis->add_animator<vec_animator>(start_frame,
                                       axis_end_frame,
                                       &x_axis->end,
                                       x_start, x_end);

    content->add_animator<vector_animator<vec>>(axis_end_frame,
                                                curve_end_frame,
                                                0, vec_points,
                                                &content->points);

    world.add_object(title_box, 0);
    world.add_object(y_axis, 0);
    world.add_object(x_axis, 0);
    world.add_object(content, 0);
}

void lefty_vis(camera& cam, scene& world) {
    vec res = vec{1200,600};
    vec origin = {50, 710};

    std::vector<double> GM_points = GM_dist(100, res, 0.086);
    //std::vector<vec> GM_points = GM_dist(100, res, 0.086);

    plot(GM_points, "temp title", res, origin, 50, 150, world);

    for (int i = 0; i < 250; ++i) {
        world.update(i);
        cam.encode_frame(world);
    }
}

int main() {
    camera cam(vec{1280, 720});
    scene world = scene({1280, 720}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 50);

    lefty_vis(cam, world);

    cam.clear_encoders();

    return 0;
}
