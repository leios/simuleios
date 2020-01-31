/*-------------sierpinsky.cpp-------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o sierpinsky sierpinsky.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

          Props to Kroppeb for help with incrementing trytes

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <gathvl/camera.h>
#include <gathvl/scene.h>
#include <utility>

typedef unsigned long long ll;
typedef std::pair<ll, ll> tryte;

tryte increment(tryte number){
    ll inc = 1;
    while(inc){
        ll for2 = number.first & inc;
        ll overf = number.second & inc;
        number.first ^= inc ^ overf;
        number.second ^= for2 ^ overf;
        inc = overf << 1;
    }
    return number;
}

void print_tryte(tryte number, int trits){
    for(int i = 1 << (trits-1); i; i >>= 1){
        if(number.first & i){
            std::cout << "1";
        }else if(number.second & i){
            std::cout << "2";
        }else{
            std::cout << "0";
        }
    }
    std::cout << std::endl;
}

vec convert(tryte number, ll scale, vec B, vec C){
    return (number.first * B + number.second * C) / scale;
}

vec hutchinson(vec init, std::vector<vec> triangle, size_t i){

    vec out;
    switch(i){
        case 0:
            out = (init + triangle[i])/2.0;
            break;
        case 1:
            out = (init + triangle[i])/2.0;
            break;
        case 2:
            out = (init + triangle[i])/2.0;
            break;
    }

    return out;
}

void sierpinski_hutchinson(camera& cam, scene& world, int n, int bin_size){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color gray = {0.5,0.5,0.5,1};

    tryte value (0,0);
    for(int i = 0; i < 3*3*3*3 ; i++){
        print_tryte(value,4);
        value = increment(value);
    }

    int size = 4;
    vec check = convert(value, (2 << (size -1 )), vec(0,1), vec(1,1));

    std::cout << check.x << '\t' << check.y << '\n';

    std::vector<vec> triangle_pts = {{0,0},{0.5,1},{1,0}};
    std::vector<std::shared_ptr<ellipse>> triangle(3);

    // Three triangle points
    for (int i = 0; i < triangle.size(); ++i){
        vec loc = {world.size.x*0.5 - 400 + 800*triangle_pts[i].x,
                   world.size.y*0.5 + 400 - 800*triangle_pts[i].y};
        triangle[i] = std::make_shared<ellipse>(loc, vec{0,0}, 0, 1);
        triangle[i]->add_animator<vec_animator>(0+i*10,30+i*10,
                                                &triangle[i]->size,
                                                vec{0,0}, vec{5,5});
    }

    // Adding elements to world
    world.add_layer();
    world.add_layer();
    for (int i = 0; i < 3; ++i){
        world.add_object(triangle[i], 2);
    }

    for (int i = 0; i < 100; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();


}

void sierpinski_chaos(camera& cam, scene& world, int n, int bin_size){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color gray = {0.5,0.5,0.5,1};

    std::vector<vec> triangle_pts = {{0,0},{0.5,1},{1,0}};
    std::vector<std::shared_ptr<ellipse>> triangle(3);

    // Three triangle points
    for (int i = 0; i < triangle.size(); ++i){
        vec loc = {world.size.x*0.5 - 400 + 800*triangle_pts[i].x,
                   world.size.y*0.5 + 400 - 800*triangle_pts[i].y};
        triangle[i] = std::make_shared<ellipse>(loc, vec{0,0}, 0, 1);
        triangle[i]->add_animator<vec_animator>(0+i*10,30+i*10,
                                                &triangle[i]->size,
                                                vec{0,0}, vec{5,5});
    }

    // Adding elements to world
    world.add_layer();
    world.add_layer();
    for (int i = 0; i < 3; ++i){
        world.add_object(triangle[i], 2);
    }


    // First, generate random point.

    vec pt = {rand() % 10000 * 0.0001, rand() % 10000 * 0.0001};

    //std::vector<std::share_ptr<ellipse>> balls;
    // Implementing Sierpinski triangle
    for (int i = 0; i < n; ++i){
        double rnd = rand() % 10000 * 0.0001;
        if (rnd <= 0.33){
            pt = {0.5*(triangle_pts[0].x + pt.x),
                  0.5*(triangle_pts[0].y + pt.y)};
        }
        else if (rnd > 0.33 && rnd <= 0.66){
            pt = {0.5*(triangle_pts[1].x + pt.x),
                  0.5*(triangle_pts[1].y + pt.y)};
        }
        else{
            pt = {0.5*(triangle_pts[2].x + pt.x),
                  0.5*(triangle_pts[2].y + pt.y)};
        }

        if (i > 20){
            vec loc = {world.size.x*0.5 - 400 + 800*pt.x,
                       world.size.y*0.5 + 400 - 800*pt.y};

            color ball_color = {1-((double)n-i)/n,0,1-(double)i/n,1};
            auto ball = std::make_shared<ellipse>(ball_color,
                                                  loc, vec{0,0}, 0, 1);
            ball->add_animator<vec_animator>(30+floor(i/bin_size)-20,
                                             60+floor(i/bin_size)-20,
                                             &ball->size,
                                             vec{0,0}, vec{2,2});
            world.add_object(ball,1);
        }
    }


    for (int i = 0; i < 1200; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

int main(){

    camera cam(vec{1920, 1080});
    scene world = scene({1920, 1080}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    sierpinski_hutchinson(cam, world, 20000, 100);
    //sierpinski_chaos(cam, world, 20000, 100);
    cam.clear_encoders();
}
