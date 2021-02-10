/*------------slider.cpp-------------------------------------------------------/
*
* Purpose: Simple slider in GathVL
*
*   Notes: Compilation instructions:
*              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o slider slider.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <vector>

#include "include/camera.h"
#include "include/scene.h"

vec find_ball_location(int frame, int start_frame, int end_frame,
                       double radius){

    double angle = 2*M_PI*((double)(frame-start_frame)/(end_frame-start_frame));

    return vec(radius*cos(angle), radius*sin(angle));
}

void rotation_scene(camera& cam, scene& world, int end_frame){
    color axis_color = {0.5, 0.5, 0.5, 1};
    color angle_color = {1,0,1,1};
    color ball_color = {0,0,1,1};

    vec origin = {world.size.x/2, world.size.y/2};
    vec ball_size = {world.size.x/20, world.size.x/20};

    auto x_axis = std::make_shared<line>(axis_color,
                                         vec(0, origin.y), vec(0, origin.y));
    auto y_axis = std::make_shared<line>(axis_color,
                                         vec(origin.x, 0), vec(origin.x, 0));
    auto angle = std::make_shared<line>(ball_color,
                                        origin, origin);

    auto unit_circle = std::make_shared<arc>(angle_color, origin,
                                             vec(world.size.x/4,world.size.x/4),
                                             vec(0,0));

    auto ball = std::make_shared<ellipse>(ball_color,
                                          origin+vec(world.size.x/4,0),
                                          vec(0,0), 0, true);

    x_axis->add_animator<vec_animator>(0,30, &x_axis->end,
                                       vec(0, origin.y),
                                       vec(world.size.x, origin.y));
    y_axis->add_animator<vec_animator>(0,30, &y_axis->end,
                                       vec(origin.x, 0),
                                       vec(origin.x, world.size.y));
    angle->add_animator<vec_animator>(0,30, &angle->end,
                                      origin,
                                      origin + vec(world.size.x/4,0));
    unit_circle->add_animator<vec_animator>(0,30, &unit_circle->angles,
                                            vec(0,0), vec(-2*M_PI,0));
    ball->add_animator<vec_animator>(0,30, &ball->size,
                                     vec(0,0), ball_size);

    world.add_layer();
    world.add_layer();

    world.add_object(x_axis, 0);
    world.add_object(y_axis, 0);
    world.add_object(angle, 1);
    world.add_object(unit_circle, 1);
    world.add_object(ball, 2);

    end_frame += 120;

    for (int i = 0; i < end_frame; ++i){

        if (i > 60 && i <= end_frame - 60){
            vec ball_loc = find_ball_location(i, 60, end_frame-60,
                                              world.size.x/4);

            ball->location = origin + ball_loc;
            angle->end = origin + ball_loc;
                                                
        }
        world.update(i);
        cam.encode_frame(world);
    }

}

void slider_scene(camera& cam, scene& world, int end_frame){

    color slider_color = {1,0,1,1};
    color ball_color = {0,0,1,1};

    double width = world.size.y*0.02;

    vec origin = {world.size.x/2, world.size.y/2};

    // Creating dial and ball
    auto dial = std::make_shared<rectangle>(slider_color,
                                            origin - vec(world.size.x/4, width),
                                            vec{0,0}, 0, true);

    auto ball = std::make_shared<ellipse>(ball_color, origin,
                                          vec{0,0}, 0, true);

    // adding entrance animations
    dial->add_animator<vec_animator>(0, 30, &dial->size,
                                     vec(0,0),
                                     vec(world.size.x/2, width*2));


    ball->add_animator<vec_animator>(0, 30, &ball->size, vec{0,0},
                                     vec{world.size.x/8, world.size.x/8});

    ball->add_animator<vec_animator>(60, 60+end_frame/4, &ball->location,
                                     origin, origin - vec(world.size.x/4,0));

    ball->add_animator<vec_animator>(60 + end_frame/4,
                                     60 + 3*end_frame/4, &ball->location,
                                     origin - vec(world.size.x/4,0),
                                     origin + vec(world.size.x/4,0));

    ball->add_animator<vec_animator>(60 + 3*end_frame/4,
                                     60 + end_frame, &ball->location,
                                     origin + vec(world.size.x/4,0),
                                     origin);

    world.add_layer();
    world.add_object(ball, 1);
    world.add_object(dial, 0);

    end_frame += 120;

    for (int i = 0; i < end_frame; ++i){
        world.update(i);
        cam.encode_frame(world);
    }
}

void numeric_scene(camera& cam, scene& world, int end_frame, double offset){

    vec origin = {world.size.x/2, world.size.y/2};

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << offset;
    std::string number = ss.str();

    auto numbers = std::make_shared<text>(origin, 50, number);

    world.add_object(numbers, 0);

    end_frame += 60;

    for (int i = 0; i < end_frame; ++i){

        double numeric = offset;
        if (i > 30 && i <= end_frame-30){

            if (i - 30 <= (end_frame-60)/4){
                numeric = offset - 0.5*(i-30)/((end_frame-60)/4);
            }
            else if (i-30 > (end_frame-60)/4 && i-30 <= 3*(end_frame-60)/4){
                numeric = offset - 0.5
                          + (i-30 - (end_frame-60)/4)/((end_frame-60)/2.0);
            }
            else {
                numeric = offset + 0.5
                          - 0.5*(i-30 - 3*(end_frame-60)/4)/((end_frame-60)/4);
            }

            std::cout << numeric << '\n';
        }

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << numeric;
        std::string number = ss.str();

        numbers->str = number;

        world.update(i);
        cam.encode_frame(world);
    }

}

int main() {
    camera cam(vec{500, 500});
    scene world = scene({500, 500}, {0, 0, 0, 1});
    //world.bg_clr = {1,1,1,1};

    //cam.add_encoder<png_encoder>();
    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);

    //slider_scene(cam, world, 100);
    //rotation_scene(cam, world, 100);
    numeric_scene(cam, world, 100, 1.0);

    cam.clear_encoders();

    return 0;
}
