/*-------------chaos.cpp------------------------------------------------------//

 Purpose: To provide a simple animation of a chaos game for various fractals

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o main chaos.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <gathvl/camera.h>
#include <gathvl/scene.h>
#include <random>

vec barnsley_tree_op(vec point, double rnd){
    vec out;
    if (rnd <= 0.02){
        out.x = 0.03*point.x;
        out.y = 0.1*point.y;
    }
    else if (rnd > 0.02 && rnd <= 0.62){
        out.x = point.x*0.85;
        out.y = 0.85*point.y + 1.5;
    }
    else if (rnd > 0.62 && rnd <= 0.72){
        out.x = point.x*0.8;
        out.y = 0.8*point.y + 1.5;
    }
    else if (rnd > 0.72 && rnd <= 0.79){
        out.x = point.x*0.2 - 0.08*point.y;
        out.y = point.x*0.15 + 0.22*point.y + .85;
    }
    else if (rnd > 0.79 && rnd <= 0.86){
        out.x = -point.x*0.2 + 0.08*point.y;
        out.y = point.x*0.15 + 0.22*point.y + .85;
    }
    else if (rnd > 0.86 && rnd <= 0.93){
        out.x = point.x*0.25 - 0.1*point.y;
        out.y = point.x*0.12 + 0.25*point.y + .4;
    }
    else{
        out.x = -point.x*0.2 + 0.1*point.y;
        out.y = point.x*0.12 + 0.2*point.y + 0.4;
    } 
    return out;
}


vec barnsley_op(vec point, double rnd){
    vec out;
    if (rnd < 0.01){
        out.x = 0;
        out.y = point.y * 0.16;
    }
    else if (rnd > 0.01 && rnd <= 0.86){
        out.x = point.x*0.85 + 0.04*point.y;
        out.y = -point.x*0.04 + 0.85*point.y + 1.6;
    }
    else if (rnd > 0.86 && rnd <= 0.93){
        out.x = point.x*0.2 - 0.26*point.y;
        out.y = point.x*0.23 + 0.22*point.y + 1.6;
    }
    else{
        out.x = -point.x*0.15 + 0.28*point.y;
        out.y = point.x*0.26 + 0.24*point.y + 0.44;
    } 
    return out;
}

vec sierpinsky_op(vec point, vec A, vec B, vec C, size_t rnd){

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

vec square_op(vec point, vec A, vec B, vec C, vec D, size_t rnd){

    vec E = (A+B+C+D)/4;
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

void animate_hutchinson(camera& cam, scene& world, size_t depth){

    color black = {0,0,0,1};
    color white = {1,1,1,1};

    std::shared_ptr<ellipse> up;
    std::shared_ptr<ellipse> left;
    std::shared_ptr<ellipse> right;

    // Adding elements to world
    world.add_layer();
    world.add_layer();

    vec loc_up = {0,0};
    vec loc_left = {0,0};
    vec loc_right = {0,0};
    vec A = {-1,-1};
    vec B = {1,-1};
    vec C = {0,1};

    vec scaled_A = vec{((A.x+1)/2)*world.size.x,
                       ((1-A.y)/2)*world.size.y};
    vec scaled_B = vec{((B.x+1)/2)*world.size.x,
                       ((1-B.y)/2)*world.size.y};
    vec scaled_C = vec{((C.x+1)/2)*world.size.x,
                       ((1-C.y)/2)*world.size.y};

    up = std::make_shared<ellipse>(white, scaled_C, vec{5,5}, 0, true);
    left = std::make_shared<ellipse>(white, scaled_A, vec{5,5}, 0, true);
    right = std::make_shared<ellipse>(white, scaled_B, vec{5,5}, 0, true);

    world.add_object(up, 1);
    world.add_object(left, 1);
    world.add_object(right, 1);

    for (size_t i = 0; i < depth; ++i){
        for (size_t j = 0; j < 3; ++j){
            loc_up = sierpinsky_op(C, A, B, C, j);
            loc_right = sierpinsky_op(A, A, B, C, j);
            loc_left = sierpinsky_op(B, A, B, C, j);

            vec scaled_up = vec{((loc_up.x+1)/2)*world.size.x,
                                ((1-loc_up.y)/2)*world.size.y};
            vec scaled_right = vec{((loc_right.x+1)/2)*world.size.x,
                                   ((1-loc_right.y)/2)*world.size.y};
            vec scaled_left = vec{((loc_left.x+1)/2)*world.size.x,
                                  ((1-loc_left.y)/2)*world.size.y};

            up = std::make_shared<ellipse>(white, scaled_up, vec{5,5}, 0, true);
            left = std::make_shared<ellipse>(white, scaled_left,
                                             vec{5,5}, 0, true);
            right = std::make_shared<ellipse>(white, scaled_right,
                                             vec{5,5}, 0, true);

            world.add_object(up, 1);
            world.add_object(left, 1);
            world.add_object(right, 1);

        }
    }

    for (int i = 0; i < 100; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();


}

void animate_chaos(camera& cam, scene& world, size_t n){
    color black = {0,0,0,1};
    color white = {1,1,1,1};

    //std::vector< std::shared_ptr<ellipse> > points(n);
    std::shared_ptr<ellipse> circle;

    // Adding elements to world
    world.add_layer();
    world.add_layer();

    vec loc = {0,0};
    vec A = {-1,-1};
    vec B = {1,-1};
    vec C = {0,1};

    //vec A = {-1,-1};
    //vec B = {-1,1};
    //vec C = {1,1};
    //vec D = {1,-1};

    for (size_t i = 0; i < n; ++i){
        //loc = sierpinsky_op(loc, A, B, C, rand()%3);
        loc = barnsley_tree_op(loc, rand()%1000 / (double)1000);
        //std::cout << loc.x << '\t' << loc.y << '\n';
        //loc = square_op(loc, A, B, C, D, rand()%5);
        vec scaled_loc = vec{((loc.x+1)/2)*world.size.x,
                             ((11-loc.y)/11)*world.size.y};
        //vec scaled_loc = vec{((loc.x+1)/2)*world.size.x,
        //                     ((1-loc.y)/2)*world.size.y};
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

    camera cam(vec{720, 720});
    scene world = scene({720, 720}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    animate_chaos(cam, world, 10000);
    //animate_hutchinson(cam, world, 1);
    cam.clear_encoders();
}
