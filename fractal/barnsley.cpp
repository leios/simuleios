/*-------------barnsley.cpp---------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o barnsley barnsley.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
          For all affine transforms, we are storing them as:
              [0 1]   [2]
              [3 4] + [5]
    TODO: Visual of pregenerated barnsley fern doing each transform

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <gathvl/camera.h>
#include <gathvl/scene.h>

vec apply_affine(vec point, double* params){
    point = {point.x*params[0] + point.y*params[1] + params[2],
             point.x*params[3] + point.y*params[4] + params[5]};
    return point;
    
}

vec apply_barnsley(vec point, int bid){
    switch(bid){
        case 0:
            point = {0.0,
                     0.16*point.y};
            break;
        case 1:
            point = {point.x*0.85 + point.y*0.04,
                     -0.04*point.x + 0.85*point.y + 1.6};
            break;
        case 2:
            point = {point.x*0.2 - point.y*0.26,
                     0.23*point.x + 0.22*point.y + 1.6};
            break;
        case 3:
            point = {-point.x*0.15 + point.y*0.28,
                     0.26*point.x + 0.24*point.y + .44};
            break;
        }

    return point;
}

// This function will create a barnsley fern and then change the fern according
//     to the selected function number and affine transform within that function
//     fx_num is the chosen barnsley function (4 possible)
//     a_num is the affine transform within that function (6 possible)
void barnsley_twiddle(camera& cam, scene& world, int n, int fx_num, int a_num){

    int frame_num = 600;
    std::vector<std::shared_ptr<ellipse>> balls(n);
    world.add_layer();
    world.add_layer();

    double twiddle_val = 0;

    // Create function set
    double fx_set[4][6] = {{0.00,0.00,0.00,0.00,0.16,0.00},
                           {0.85, 0.04, 0.00, -0.04, 0.85, 1.60},
                           {0.2, -0.26, 0.00, 0.23, 0.22, 1.6},
                           {-0.15, 0.28, 0.00, 0.26, 0.24, 0.44}};

    double modified_element = fx_set[fx_num][a_num];

    double twiddle_max = 0.5;
    if (1-abs(fx_set[fx_num][a_num]) < 0.5){
        twiddle_max = 1-abs(fx_set[fx_num][a_num]);
    }

    // Do chaos game
    // Implementing barnsley fern
    for (int i = 0; i < frame_num; ++i){
        vec pt = vec{0,0};
        if (i < frame_num/4){
            twiddle_val = i * twiddle_max / (0.25*frame_num);
        }
        else if (i >= 0.25 * frame_num && i < 0.5 * frame_num){
            twiddle_val = twiddle_max
                          - (i-0.25*frame_num) * twiddle_max / (0.25*frame_num);
        }
        else if (i >= 0.5 * frame_num && i < 0.75 * frame_num){
            twiddle_val = 0.0
                          - (i-0.5*frame_num) * twiddle_max / (0.25*frame_num);
        }
        else if (i >= 0.75 * frame_num){
            twiddle_val = -twiddle_max
                          + (i-0.75*frame_num) * twiddle_max / (0.25*frame_num);
        }
        //std::cout << twiddle_val << '\n';

        fx_set[fx_num][a_num] = modified_element + twiddle_val;
        if (fx_set[fx_num][a_num] > 1){
            fx_set[fx_num][a_num] = 1;
        }
        std::cout << i << '\t' << fx_set[fx_num][a_num] << '\n';
        
        srand(1337);
        for (int j = 0; j < n; ++j){
            double rnd = rand() % 10000 * 0.0001;
            if (rnd <= 0.01){
                pt = apply_affine(pt, fx_set[0]);
            }
            else if (rnd > 0.01 && rnd <= 0.86){
                pt = apply_affine(pt, fx_set[1]);
            }
            else if (rnd > 0.86 && rnd <= 0.93){
                pt = apply_affine(pt, fx_set[2]);
            }
            else{
                pt = apply_affine(pt, fx_set[3]);
            }

            vec loc = {world.size.x*0.5 + 200*pt.x,
                       world.size.y*0.5 + 500 - 100*pt.y};

            color ball_color = {0,1-((double)n-j)/n,1-(double)j/n,1};
            auto ball = std::make_shared<ellipse>(ball_color,
                                                  loc, vec{2,2}, 0, 1);
            world.add_object(ball,1);
        }


        world.update(i);
        cam.encode_frame(world);
        world.clear_layer(1);
    }


}

// TODO: modify such that we perform the stem transformation, then the right and
//       left leaf transforms, and then the successive leaves upwards.
void barnsley_quadtransform(camera& cam, scene& world, int n){
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

    int order[4] = {1,2,3,1};
    for (int j = 0; j < 4; ++j){
        for (int i = 0; i < pts.size(); ++i){
            vec loc_prev = {world.size.x*0.5 + 200*pts[i].x,
                            world.size.y*0.5 + 200 - 200*pts[i].y};

            pts[i] = apply_barnsley(pts[i],order[j]);

            vec loc_after = {world.size.x*0.5 + 200*pts[i].x,
                             world.size.y*0.5 + 200 - 200*pts[i].y};
            balls[i]->add_animator<vec_animator>(60+j*60,
                                                 120+j*60,
                                                 &balls[i]->location,
                                                 loc_prev, loc_after);
            world.add_object(balls[i], 1);
        }
    }
    for (int i = 0; i < 600; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();

}

// This function creates a barnsley fern and uses the chosen
//     transformation from the barnsley fern to get an idea what each one does
void barnsley_fern_transform(camera& cam, scene& world, int n, int ttype){

    std::vector<vec> pts(n);
    std::vector<std::shared_ptr<ellipse>> balls(n), new_balls(n);
    world.add_layer();
    world.add_layer();

    int bf = 0;
    for (int i = 0; i < pts.size(); ++i){
        if (i == 0){
            pts[0] = {0,0};
        }
        else {
            double rnd = (double)(rand() % 1000) / 1000;
            if (rnd <= 0.01){
                bf = 0;
            }
            else if (rnd > 0.01 && rnd <= 0.86){
                bf = 1;
            }
            else if (rnd > 0.86 && rnd <= 0.93){
                bf = 2;
            }
            else{
                bf = 3;
            }

            pts[i] = apply_barnsley(pts[i-1], bf);
        }

        vec loc = {world.size.x*0.5 + 200*pts[i].x,
                   world.size.y*0.5 + 500 - 100*pts[i].y};
        color ball_color = {0,1-((double)n-i)/n,1-(double)i/n,1};

        balls[i] = std::make_shared<ellipse>(ball_color, loc, vec{0,0}, 0, 1);
        balls[i]->add_animator<vec_animator>(0,30,&balls[i]->size,
                                             vec{0,0}, vec{2,2});
        new_balls[i] = std::make_shared<ellipse>(ball_color,
                                                 loc, vec{0,0}, 0, 1);
        new_balls[i]->add_animator<vec_animator>(0,30,&new_balls[i]->size,
                                                 vec{0,0}, vec{2,2});
    }

    for (int i = 0; i < pts.size(); ++i){
        vec loc_prev = {world.size.x*0.5 + 200*pts[i].x,
                        world.size.y*0.5 + 500 - 100*pts[i].y};
        pts[i] = apply_barnsley(pts[i],ttype);
        vec loc_after = {world.size.x*0.5 + 200*pts[i].x,
                         world.size.y*0.5 + 500 - 100*pts[i].y};
        new_balls[i]->add_animator<vec_animator>(60,120,&new_balls[i]->location,
                                                 loc_prev, loc_after);
        new_balls[i]->add_animator<color_animator>(60,120,&new_balls[i]->clr,
                                                   new_balls[i]->clr, 
                                                   color{0.5,0,0,1});
        world.add_object(balls[i], 1);
        world.add_object(new_balls[i], 2);
    }

    for (int i = 0; i < 200; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}


// This function creates a random distribution of points and uses the chosen
//     transformation from the barnsley fern to get an idea what each one does
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
        pts[i] = apply_barnsley(pts[i],ttype);
        vec loc_after = {world.size.x*0.5 + 200*pts[i].x,
                         world.size.y*0.5 + 200 - 200*pts[i].y};
        balls[i]->add_animator<vec_animator>(60,120,&balls[i]->location,
                                             loc_prev, loc_after);
        world.add_object(balls[i], 1);
    }

    for (int i = 0; i < 200; ++i){
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
            pt = apply_barnsley(pt, 0);
        }
        else if (rnd > 0.01 && rnd <= 0.86){
            pt = apply_barnsley(pt, 1);
        }
        else if (rnd > 0.86 && rnd <= 0.93){
            pt = apply_barnsley(pt, 2);
        }
        else{
            pt = apply_barnsley(pt, 3);
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
    double a_test[1][6] = {{1,2,3,4,5,6}};
    world.bg_clr = {1,1,1,1};

    vec pt = vec{1,1};
    pt = apply_affine(pt, a_test[0]);

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    //barnsley(cam, world, 100000, 1000);
    //barnsley_quadtransform(cam, world, 1000);
    //barnsley_transform(cam, world, 1000, 0);
    barnsley_fern_transform(cam, world, 100000, 3);
    //barnsley_twiddle(cam, world, 10000, 1, 0);
    cam.clear_encoders();
/*
    std::string filebase = "/tmp/video";
    std::string fileend = ".mp4";
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 6; ++j){
            std::string filename = filebase + std::to_string(i)
                                   + std::to_string(j) + fileend;
            std::cout << filename << '\n';
            cam.add_encoder<video_encoder>(filename, cam.size, 60);
            barnsley_twiddle(cam, world, 10000, i, j);
            cam.clear_encoders();
        }
    }
*/
}
