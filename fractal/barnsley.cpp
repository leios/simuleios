/*-------------barnsley.cpp---------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o barnsley barnsley.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <gathvl/camera.h>
#include <gathvl/scene.h>

// TODO: modify such that we perform the stem transformation, then the right and
//       left leaf transforms, and then the successive leaves upwards.
void barnsley_transform(camera& cam, scene& world, int n, int ttype){

    std::vector<vec> pts(n);
    std::vector<std::shared_ptr<ellipse>> balls(n);
    world.add_layer();
    world.add_layer();

    for (int i = 0; i < pts.size(); ++i){
        pts[i].x = (rand() % 10000 * 0.0001)*2 - 1;
        pts[i].y = (rand() % 10000 * 0.0001)*2 - 1;
        vec loc = {world.size.x*0.5 + 200*pts[i].x,
                   world.size.y*0.5 + 200 - 200*pts[i].y};
        color ball_color = {0,1-((double)n-i)/n,1-(double)i/n,1};

        balls[i] = std::make_shared<ellipse>(ball_color, loc, vec{0,0}, 0, 1);
        balls[i]->add_animator<vec_animator>(0,30,&balls[i]->size,
                                             vec{0,0}, vec{5,5});
    }

    for (int i = 0; i < pts.size(); ++i){
        vec loc_prev = {world.size.x*0.5 + 200*pts[i].x,
                        world.size.y*0.5 + 200 - 200*pts[i].y};
        switch(ttype){
            case 0:
                pts[i] = {0.0,
                          0.16*pts[i].y};
                break;
            case 1:
                pts[i] = {pts[i].x*0.84 + pts[i].y*0.04,
                          -0.04*pts[i].x + 0.84*pts[i].y + 1.6};
                break;
            case 2:
                pts[i] = {pts[i].x*0.2 - pts[i].y*0.26,
                          0.23*pts[i].x + 0.22*pts[i].y + 1.6};
                break;
            case 3:
                pts[i] = {-pts[i].x*0.15 + pts[i].y*0.28,
                          0.26*pts[i].x + 0.24*pts[i].y + .44};
                break;
        }
        vec loc_after = {world.size.x*0.5 + 200*pts[i].x,
                         world.size.y*0.5 + 200 - 200*pts[i].y};
        balls[i]->add_animator<vec_animator>(60,120,&balls[i]->location,
                                             loc_prev, loc_after);
        world.add_object(balls[i], 1);
    }
    for (int i = 0; i < 600; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

void barnsley(camera& cam, scene& world, int n, int bin_size){
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
    //barnsley_transform(cam, world, 1000, 3);
    cam.clear_encoders();
}
