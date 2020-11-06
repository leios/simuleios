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

#include "include/camera.h"
#include "include/scene.h"

std::vector<vec> normalize(std::vector<vec> points, double factor){
    double max_value = 0.0;
    for (int i = 0; i < points.size(); ++i){
         if (points[i].y > max_value){
             max_value = points[i].y;
         }
    }

    std::cout << max_value << '\n';

    for (int i = 0; i < points.size(); ++i){
        points[i].y = points[i].y * factor / max_value;
    }

    return points;
}

std::vector<vec> GM_dist(int res, vec dim, double beta){
    std::vector<vec> GM_points(res);
    for (int i = 0; i < res; ++i) {
        GM_points.emplace_back(i * dim.x/res, std::exp(i * beta));
    }

    return normalize(GM_points, dim.y);
    //return GM_points;
}

void lefty_vis(camera& cam, scene& world) {
    auto title = std::make_shared<text>(vec{0, 720}, 50, "GathVL Test");

    auto y_axis = std::make_shared<line>(vec{50, 10}, vec{50, 610});
    auto x_axis = std::make_shared<line>(vec{1250, 610}, vec{50, 610});

    vec origin = {50, 610};

    std::vector<vec> GM_points = GM_dist(100, vec{1200,600}, 0.086);

    auto GM_curve =
        std::make_shared<curve>(std::vector<vec>(), origin);

    GM_curve->add_animator<vector_animator<vec>>(0, 200, 0, GM_points,
                                                 &GM_curve->points);

    world.add_object(title, 0);
    world.add_object(y_axis, 0);
    world.add_object(x_axis, 0);
    world.add_object(GM_curve, 0);

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
