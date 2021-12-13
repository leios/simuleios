/*------------warp_divergence.cpp----------------------------------------------/
*
* Purpose: Simple visualization script warp divergence
*
*   Notes: Compilation instructions:
*              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o gauss_vis gauss_vis.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <string>

#include "include/camera.h"
#include "include/scene.h"

void warp_divergence(camera& cam, scene &world, color &clr1, color &clr2){

    world.add_layer();
    world.add_layer();
    color line_clr = {0,0,0,1};
    int text_size = 30;

    vec origin = {world.size.x/2, world.size.y/2};
    int num_lines = 31;

    vec rect_size = {world.size.x*0.9, world.size.y*0.1};
    vec rect_loc = origin - rect_size*0.5;
    
    auto rec = std::make_shared<rectangle>(line_clr, rect_loc, rect_size, 0,
                                           false);

    vec start_loc = {0, origin.y-rect_size.y*0.5};
    vec end_loc = {0,origin.y+rect_size.y*0.5};

    vec ball_loc = {0,origin.y};
    vec ball_size = {world.size.y*0.015, world.size.y*0.015};

    color ball_clr = {0,0,0,0};

    double dx = rect_size.x / (num_lines + 1);
    for (size_t i = 0; i <= num_lines; ++i){
        if (i < num_lines){
            start_loc.x = rect_loc.x + dx*(i+1);
            end_loc.x = rect_loc.x + dx*(i+1);

            auto indicator = std::make_shared<line>(line_clr,
                                                    start_loc, end_loc);
            world.add_object(indicator, 1);
        }

        ball_loc.x = rect_loc.x + dx*(i+0.5);

        // Setting locations for drop-down circles
        if (i % 2 == 0){
            ball_loc.y += rect_size.y;
            ball_clr = clr1;
        }
        else{
            ball_loc.y += 2*rect_size.y;
            ball_clr = clr2;
        }

        auto ball = std::make_shared<ellipse>(ball_clr, ball_loc,
                                              ball_size, 0, true);
        world.add_object(ball, 1);

        auto connection = std::make_shared<line>(ball_clr,
                                                 vec(ball_loc.x, origin.y),
                                                 ball_loc);

        world.add_object(connection, 2);

        start_loc.x = rect_loc.x + dx*(i+1);
        end_loc.x = rect_loc.x + dx*(i+1);

        auto shade = std::make_shared<rectangle>(ball_clr,
                                                 vec(rect_loc.x + dx*(i),
                                                     rect_loc.y),
                                                 vec(rect_size.x/32,
                                                     rect_size.y),
                                                 0, true);

        world.add_object(shade, 0);

        int word_offset = text_size*ceil(log10(1+i))*0.3;
        if (i == 0){
            word_offset = text_size*0.3;
        }
        auto thread_num = std::make_shared<text>(line_clr,
                                                 vec(ball_loc.x-word_offset,
                                                     rect_loc.y-0.3*text_size),
                                                 text_size,
                                                 std::to_string(i), 0);

        world.add_object(thread_num, 0);


        // resetting locations for circles
        if (i % 2 == 0){
            ball_loc.y -= rect_size.y;
        }
        else{
            ball_loc.y -= 2*rect_size.y;
        }
    }


    //world.add_object(ball, 1);
    world.add_object(rec, 1);

    world.update(0);
    cam.encode_frame(world);

    world.clear();

}

int main(){
    camera cam(vec{1920, 1080});
    scene world = scene({1920, 1080}, {0, 0, 0, 1});

    world.bg_clr = {1,1,1,1};

    color clr1 = {1, 0.25, 1, 1};
    color clr2 = {0.25, 0.25, 1, 1};

    cam.add_encoder<png_encoder>("/tmp/");

    warp_divergence(cam, world, clr1, clr2);

    cam.clear_encoders();

    return 0;

}
