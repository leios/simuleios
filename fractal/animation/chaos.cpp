/*-------------chaos.cpp------------------------------------------------------//

 Purpose: To provide a simple animation of a chaos game for various fractals

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o main chaos.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <gathvl/camera.h>
#include <gathvl/scene.h>
#include <random>

vec sierpinsky_op(vec point, vec A, vec B, vec C){

    int rnd = rand() % 3;
    vec out;

    switch(rnd){
        case 0:
            out = (A + point)/2;
            break;
        case 1:
            out = (B + point)/2;
            break;
        case 2:
            out = (C + point)/2;
            break;
    }
    return out;
}

vec square_op(vec point, vec A, vec B, vec C, vec D){

    vec E = (A+B+C+D)/4;
    int rnd = rand() % 5;
    vec out;

    switch(rnd){
        case 0:
            out = (2*A + point)/3;
            break;
        case 1:
            out = (2*B + point)/3;
            break;
        case 2:
            out = (2*C + point)/3;
            break;
        case 3:
            out = (2*D + point)/3;
            break;
        case 4:
            out = (2*E + point)/3;
            break;
    }
    return out;
}


void animate_chaos(camera& cam, scene& world, size_t n){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color pink = {1,0,1,1};

    //std::vector< std::shared_ptr<ellipse> > points(n);
    std::shared_ptr<ellipse> circle;

    // Adding elements to world
    world.add_layer();
    world.add_layer();

    vec loc = {0,0};
    //vec A = {-1,-1};
    //vec B = {1,-1};
    //vec C = {0,1};

    vec A = {-1,-1};
    vec B = {-1,1};
    vec C = {1,1};
    vec D = {1,-1};

    for (size_t i = 0; i < n; ++i){
        //loc = sierpinsky_op(loc, A, B, C);
        loc = square_op(loc, A, B, C, D);
        vec scaled_loc = vec{((loc.x+1)/2)*world.size.x,
                             ((1-loc.y)/2)*world.size.y};
        color clr = {1-(double)i/n,0,(double)i/n,1};
        circle = std::make_shared<ellipse>(clr, scaled_loc, vec{0,0},
                                           0, true);
        circle->add_animator<vec_animator>(0+i/10,30+i/10, &circle->size, vec{0,0}, vec{2,2});
        world.add_object(circle, 1);
    }

    for (int i = 0; i < 1100; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

int main(){

    camera cam(vec{1280, 720});
    scene world = scene({1280, 720}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    animate_chaos(cam, world, 10000);
    cam.clear_encoders();
}
