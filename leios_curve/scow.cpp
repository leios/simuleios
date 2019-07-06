/*-------------scow.cpp-------------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o main scow.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <gathvl/camera.h>
#include <gathvl/scene.h>

void draw_scow(camera& cam, scene& world){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color pink = {1,0,1,1};
    // Body is an outline, we'll fill it in later
    auto body = std::make_shared<arc>(vec{world.size.x*0.5,world.size.y*0.5}, 
                                      vec{world.size.y*0.25,world.size.y*0.25},
                                      vec{0,0});

    body->add_animator<vec_animator>(0, 60, &body->angles,
                                     vec{0*M_PI, 0*M_PI},
                                     vec{0*M_PI,1.9*M_PI});

    // We only have 3 legs
    std::vector< std::shared_ptr<rectangle> > legs(3);
    for (int i = 0; i < 3; ++i){
        vec loc;
        double rot; 

        // front right leg
        if (i == 0){
            loc = {world.size.x*0.5 + world.size.y*0.175,
                   world.size.y*0.5 + world.size.y*0.225};
            rot = -0.2*M_PI;
        }

        // front left leg
        else if (i == 1){
            loc = {world.size.x*0.5 + world.size.y*0.225,
                   world.size.y*0.5 + world.size.y*0.175};
            rot = -0.3*M_PI;
        }

        // back leg
        else {
            loc = {world.size.x*0.5 - world.size.y*0.2,
                   world.size.y*0.5 + world.size.y*0.2};
            rot = 0.25*M_PI;
        }
        vec size = {0,0};
        legs[i] = std::make_shared<rectangle>(loc, size, rot, 1);
        legs[i]->add_animator<vec_animator>(0, 60, &legs[i]->size,
                                            vec{world.size.y*0.0125,0},
                                            vec{world.size.y*0.0125,
                                                world.size.y*0.02});
    }

    // Tail is a weird polygon
    // The fill is set to 0 for now, but we will fix it soon for a polygon
    vec origin = {world.size.x*0.35, world.size.y*0.35};
    std::vector<vec> tail_points_init = {{world.size.y*0.01 + origin.x,
                                          world.size.y*0.01 + origin.y},
                                         {world.size.y*0.01 + origin.x,
                                          world.size.y*0.01 + origin.y},
                                         {-world.size.y*0.01 + origin.x,
                                          world.size.y*0.01 + origin.y},
                                         {-world.size.y*0.01 + origin.x,
                                          world.size.y*0.01 + origin.y}};
    std::vector<vec> tail_points_final = {{world.size.y*0.01 + origin.x,
                                           world.size.y*0.01 + origin.y},
                                          {world.size.y*0.01 + origin.x,
                                           -world.size.y*0.02 + origin.y},
                                          {-world.size.y*0.02 + origin.x,
                                           -world.size.y*0.02 + origin.y},
                                          {-world.size.y*0.01 + origin.x,
                                           world.size.y*0.01 + origin.y}};
    auto tail = std::make_shared<polygon>(tail_points_init, 1, -0.75, origin);

    tail->add_animator<vec_animator>(0, 60, &tail->points[1],
                                     tail_points_init[1], tail_points_final[1]);
    tail->add_animator<vec_animator>(0, 60, &tail->points[2],
                                     tail_points_init[2], tail_points_final[2]);

    // Working on the head
    vec head_origin = {world.size.x * 0.625, world.size.y * 0.45};
    std::vector<vec> head_points = {{world.size.y*0.055 + head_origin.x,
                                     world.size.y*0.06 + head_origin.y},
                                    {world.size.y*0.045 + head_origin.x,
                                     -world.size.y*0.05 + head_origin.y},
                                    {-world.size.y*0.04 + head_origin.x,
                                     -world.size.y*0.05 + head_origin.y},
                                    {-world.size.y*0.04 + head_origin.x,
                                     world.size.y*0.06 + head_origin.y}};

    auto head_bg = std::make_shared<polygon>(head_points, 2, -.2, head_origin);

    // TODO: fix head outline and color animation for head_bg
    auto head_outline = std::make_shared<curve>(std::vector<vec>(),
                                                //head_origin);
                                                vec(0,0));
/*
    head_bg->add_animator<vec_animator>(0,60,&head_bg->clr,
                                        black, white);

    head_outline->add_animator<vector_animator<vec>>(0,60, 0, head_points,
                                                     &head_outline->points);
*/

    // Adding elements to world
    world.add_layer();
    world.add_layer();
    world.add_object(body, 1);
    for (int i = 0; i < 3; ++i){
        world.add_object(legs[i], 2);
    }
    world.add_object(head_outline, 2);
    world.add_object(tail, 2);
    //world.add_object(head_bg, 1);

    for (int i = 0; i < 200; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

int main(){

    camera cam(vec{1280, 720});
    scene world = scene({1280, 720}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    draw_scow(cam, world);
    cam.clear_encoders();
}
