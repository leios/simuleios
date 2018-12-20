/*-------------barnsley.cpp---------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o barnsley barnsley.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <gathvl/camera.h>
#include <gathvl/scene.h>

void barnsley(camera& cam, scene& world, int n, int bin_size){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color gray = {0.5,0.5,0.5,1};

    std::vector<vec> triangle_pts = {{0,0},{0.5,1},{1,0}};
    std::vector<std::shared_ptr<ellipse>> triangle(3);

    // Adding elements to world
    world.add_layer();
    world.add_layer();

    // First, generate random point.

    vec pt = {0, 0};

    // Implementing barnsley fern
    for (int i = 0; i < n; ++i){
        double rnd = rand() % 10000 * 0.0001;
        if (rnd <= 0.01){
            pt = {0.0,
                  0.16*pt.y};
        }
        else if (rnd > 0.01 && rnd <= 0.86){
            pt = {pt.x*0.84 + pt.y*0.04,
                  -0.04*pt.x + 0.84*pt.y + 1.6};
        }
        else if (rnd > 0.86 && rnd <= 0.93){
            pt = {pt.x*0.2 - pt.y*0.26,
                  0.23*pt.x + 0.22*pt.y + 1.6};
        }
        else{
            pt = {-pt.x*0.15 + pt.y*0.28,
                  0.26*pt.x + 0.24*pt.y + .44};
        }

        vec loc = {world.size.x*0.5 + 200*pt.x,
                   world.size.y*0.5 + 500 - 100*pt.y};

        color ball_color = {0,1-((double)n-i)/n,1-(double)i/n,1};
        auto ball = std::make_shared<ellipse>(ball_color,
                                              loc, vec{0,0}, 0, 1);
        ball->add_animator<vec_animator>(30+floor(i/bin_size)-20,
                                         60+floor(i/bin_size)-20,
                                         &ball->size,
                                         vec{0,0}, vec{2,2});
        world.add_object(ball,1);
    }


    for (int i = 0; i < 600; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

int main(){

    camera cam(vec{1920, 1080});
    scene world = scene({1920, 1080}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    barnsley(cam, world, 20000, 100);
    cam.clear_encoders();
}
