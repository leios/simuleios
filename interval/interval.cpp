/*------------gauss_vis.cpp----------------------------------------------------/
*
* Purpose: Simple visualization script for interval training video
*
*   Notes: Compilation instructions:
*              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o interval interval.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>

#include "include/camera.h"
#include "include/scene.h"

void clock_scene(camera& cam, scene& world){
}

void priority_scene(camera& cam, scene& world){

    double text_size = world.size.x*0.075;
    color red = {1, 0.25, 0.25, 1};
    color green = {0.25, 1, 0.25, 1};
    color blue = {0.25, 0.25, 1, 1};

    std::vector<std::shared_ptr<rectangle>> priority_boxes(3);
    std::vector<std::shared_ptr<text>> numbers(3);

    // placing boxes in appropriate locations
    vec loc = vec{world.size.x*0.5 - world.size.x*0.125,
                  world.size.y*0.5 - world.size.y*0.125};

    vec new_loc = vec{world.size.x*0.5 - world.size.x*0.2,
                      world.size.y*0.25 - world.size.y*0.2};
    priority_boxes[0] = std::make_shared<rectangle>(red,
                                                    loc,
                                                    vec{world.size.x*0.25,
                                                        world.size.y*0.25},
                                                    0,
                                                    true);

    priority_boxes[0]->add_animator<vec_animator>(60,120,
                                                  &priority_boxes[0]->location,
                                                  loc,
                                                  new_loc);
    priority_boxes[0]->add_animator<vec_animator>(60,120,
                                                  &priority_boxes[0]->size,
                                                  vec{world.size.x*0.25,
                                                      world.size.y*0.25},
                                                  vec{world.size.x*0.4,
                                                      world.size.y*0.4});
    loc = vec{world.size.x*0.16666 - world.size.x*0.125,
              world.size.y*0.5 - world.size.y*0.125};
    priority_boxes[1] = std::make_shared<rectangle>(green,
                                                    loc,
                                                    vec{world.size.x*0.25,
                                                        world.size.y*0.25},
                                                    0,
                                                    true);

    priority_boxes[1]->add_animator<vec_animator>(60,120,
                                                  &priority_boxes[1]->location,
                                                  loc,
                                                  loc+vec{0,world.size.y*0.25}); 
    loc = vec{world.size.x*0.8333 - world.size.x*0.125,
              world.size.y*0.5 - world.size.y*0.125};
    priority_boxes[2] = std::make_shared<rectangle>(blue,
                                                    loc,
                                                    vec{world.size.x*0.25,
                                                        world.size.y*0.25},
                                                    0,
                                                    true);

    priority_boxes[2]->add_animator<vec_animator>(60,120,
                                                  &priority_boxes[2]->location,
                                                  loc,
                                                  loc+vec{0,world.size.y*0.25}); 

    // placing text in appropriate locations
    loc = vec{world.size.x*0.5 - text_size*0.35,
              world.size.y*0.5 + text_size*0.35};
    numbers[0] = std::make_shared<text>(loc, text_size, "2");
    numbers[0]->add_animator<vec_animator>(60,120,
                                           &numbers[0]->location,
                                           loc,
                                           loc-vec{0,world.size.y*0.25}); 

    loc = vec{world.size.x*0.16666 - text_size*0.35,
              world.size.y*0.5 + text_size*0.35};
    numbers[1] = std::make_shared<text>(loc, text_size, "1");
    numbers[1]->add_animator<vec_animator>(60,120,
                                           &numbers[1]->location,
                                           loc,
                                           loc+vec{0,world.size.y*0.25}); 

    loc = vec{world.size.x*0.8333 - text_size*0.35,
              world.size.y*0.5 + text_size*0.35};
    numbers[2] = std::make_shared<text>(loc, text_size, "3");
    numbers[2]->add_animator<vec_animator>(60,120,
                                           &numbers[2]->location,
                                           loc,
                                           loc+vec{0,world.size.y*0.25}); 

    world.add_layer();
    world.add_layer();
    for (int i = 0; i < 3; ++i){
        world.add_object(priority_boxes[i],0);
        world.add_object(numbers[i],1);
    }

    for (int i = 0; i < 200; ++i) {
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();

}

void shuffle_scene(camera& cam, scene& world){
}

int main() {
    camera cam(vec{1920,1080});
    scene world = scene({1920,1080}, {0, 0, 0, 1});
    //cairo_set_line_width(world.context.get(), 10);

    //cam.add_encoder<png_encoder>();
    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);

    priority_scene(cam, world);

    cam.clear_encoders();

    return 0;
}
