/*-------------sierpinski.cpp-------------------------------------------------//

 Purpose: This is a stand-alone file to draw the spherical cow logo

   Notes: Compile with
              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o sierpinski sierpinski.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL

          Props to Kroppeb for help with incrementing trytes and quits

    TODO: 1. Remove magic numbers
          2. Finish animation of hutchinson
          3. Square distribution
          4. Change all ball sizes

*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <gathvl/camera.h>
#include <gathvl/scene.h>
#include <utility>

vec midpoint(vec point1, vec point2){
    return 0.5*(point1 + point2);
}

vec two_thirds(vec point1, vec point2){
    return (point1 + 2*point2)/3.0;
}

vec apply_quit(std::string value_string, vec point,
               std::vector<vec> square_points){
    for (int i = 0; i < value_string.size(); ++i){
        char value = value_string[i];
        switch(value){
            case '0':
                point = midpoint(point, square_points[0]);
                break;
            case '1':
                point = midpoint(point, square_points[1]);
                break;
            case '2':
                point = midpoint(point, square_points[2]);
                break;
            case '3':
                point = midpoint(point, square_points[3]);
                break;
        }
    }

    return point;
}

vec apply_tryte(std::string value_string, vec point,
                std::vector<vec> triangle_points){
    for (int i = 0; i < value_string.size(); ++i){
        char value = value_string[i];
        switch(value){
            case '0':
                point = midpoint(point, triangle_points[0]);
                break;
            case '1':
                point = midpoint(point, triangle_points[1]);
                break;
            case '2':
                point = midpoint(point, triangle_points[2]);
                break;
        }
    }

    return point;
}

// TODO: Modify trytes to work with bijective base 3
typedef unsigned long long ll;
typedef std::pair<ll, ll> tryte, quit;

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

std::string save_tryte(tryte number, int trytes){
    std::string value_string = "";
    for(int i = 1 << (trytes-1); i; i >>= 1){
        if(number.first & i){
            value_string += "0";
        }else if(number.second & i){
            value_string += "1";
        }else{
            value_string += "2";
        }
    }
    return value_string;
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

quit increment_quit(quit number){
    ll inc = 1;
    while(inc){
        ll for2 = number.first & inc;
        ll overf = number.second & for2;
        number.first ^= inc;
        number.second ^= for2;
        inc = overf << 1;
    }
    return number;
}
std::string save_quit(quit number, int quits){
    std::string value_string = "";
    for(int i = 1 << (quits-1); i; i >>= 1){
        int value = (((number.second & i) > 0) << 1) |
                    (((number.first & i) > 0));
        switch(value){
            case 0:
                value_string += "3";
                break;
            case 1:
                value_string += "0";
                break;
            case 2:
                value_string += "1";
                break;
            case 3:
                value_string += "2";
                break;
        }
    }
    return value_string;
}

void print_increment(){
    tryte value (0,0);
    int level = 1;
    int diff = 3;
    for(int i = 0; i < 3*3*3*3 ; i++){
        if (i == diff){
            level += 1;
            value = {0,0};
            diff += pow(3,level);
        }
        print_tryte(value,level);
        value = increment(value);
        //std::cout << value.first << '\t' << value.second << '\n';
    }
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

void square_hutchinson(camera& cam, scene& world, int level,
                       int total_frames, bool aaa_version){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color blue  = {0,0,1,1};
    color pink  = {1,0,1,1};
    color red   = {1,0,0,1};
    color green = {0,1,0,1};
    color gray  = {0.5,0.5,0.5,1};

    color pt_clr, bg_clr;
    if (aaa_version){
        pt_clr = black;
        bg_clr = white;
    }
    else{
        pt_clr = white;
        bg_clr = black;
    }

    double text_size = 100;
    double ball_size = (level*2);
    std::vector<vec> square_pts = {{0,0},{0,1},{1,1},{1,0}};
    std::vector<vec> square_midpts = {0.5*(square_pts[0]+square_pts[1]),
                                      0.5*(square_pts[1]+square_pts[2]),
                                      0.5*(square_pts[2]+square_pts[3]),
                                      0.5*(square_pts[0]+square_pts[2]),
                                      0.5*(square_pts[3]+square_pts[0])};
    
    std::vector<std::shared_ptr<ellipse>> square(4), midsquare(5);

    int square_size = world.size.y*0.8;
    int square_offset = square_size*0.5;

    std::shared_ptr<text> label;
    vec label_offset;
    std::string letter;

    // Four square points
    for (int i = 0; i < square.size(); ++i){
        switch(i){
            case 0:
                letter = "A";
                label_offset = {0, text_size};
                break;
            case 1:
                letter = "B";
                label_offset = {0, -0.25*text_size};
                break;
            case 2:
                letter = "C";
                label_offset = {0, -0.25*text_size};
                break;
            case 3:
                letter = "D";
                label_offset = {0, text_size};
                break;
        }

        vec loc = {world.size.x*0.5 - square_offset 
                   + square_size*square_pts[i].x,
                   world.size.y*0.5 + square_offset 
                   - square_size*square_pts[i].y};
        square[i] = std::make_shared<ellipse>(pt_clr, loc, vec{0,0}, 0, 1);
        square[i]->add_animator<vec_animator>(0+i*10,30+i*10,
                                                &square[i]->size,
                                                vec{0,0}, vec{10,10});

        label = std::make_shared<text>(bg_clr, loc+label_offset, text_size,
                                       letter, 0);
        label->add_animator<color_animator>(0+i*10,30+i*10,
                                           &label->clr,
                                           bg_clr,pt_clr);
        world.add_object(label, 0);
    }
    for (int i = 0; i < midsquare.size(); ++i){
        vec loc = {world.size.x*0.5 - square_offset 
               + square_size*square_midpts[i].x,
               world.size.y*0.5 + square_offset 
               - square_size*square_midpts[i].y};
        midsquare[i] = std::make_shared<ellipse>(pt_clr, loc, vec{0,0}, 0, 1);
        midsquare[i]->add_animator<vec_animator>(60+i*10,90+i*10,
                                                &midsquare[i]->size,
                                                vec{0,0}, vec{10,10});
    }

    vec loc, parent_loc;
    auto point = std::make_shared<ellipse>(pt_clr, vec{0,0}, vec{10,10}, 0, 1);

    world.add_layer();
    world.add_layer();

    quit value (0,0);
    int tmp_level = 1;
    int diff = 4;
    std::string value_string = "", parent_string;

    int draw_frame = 120;

    // This is creating all the children
    for(int i = 0; i < ((pow(4,level)-1)/3)-1; i++){
        if (i == diff){
            tmp_level += 1;
            value = {0,0};
            diff += pow(4,tmp_level);
            draw_frame += 60;
            if (ball_size > 1){
                ball_size /= 1.4;
            }
            else{
                ball_size = 1;
            }
        }
        value = increment_quit(value);
        value_string = save_quit(value,tmp_level);
        for (int j = 0; j < 5; ++j){
            color point_clr;
            switch(j){
                case 0:
                    point_clr = red;
                    break;
                case 1:
                    point_clr = green;
                    break;
                case 2:
                    point_clr = blue;
                    break;
                case 3:
                    point_clr = gray;
                    break;
                case 4:
                    point_clr = pink;
                    break;
            }
            loc = apply_quit(value_string, square_midpts[j],
                             square_pts);
            parent_string = value_string.substr(0,value_string.size()-1);
            parent_loc = apply_quit(parent_string, square_midpts[j],
                                    square_pts);
            loc = {world.size.x*0.5 - square_offset + square_size*loc.x,
                   world.size.y*0.5 + square_offset
                   - square_size*loc.y};
            parent_loc = {world.size.x*0.5 - square_offset
                          + square_size*parent_loc.x,
                          world.size.y*0.5 + square_offset
                          - square_size*parent_loc.y};
            point = std::make_shared<ellipse>(point_clr, loc,
                                              vec{0,0}, 0, 1);
            point->add_animator<vec_animator>(draw_frame, draw_frame+30,
                                              &point->size,vec{0,0},
                                              vec{ball_size, ball_size});
            point->add_animator<vec_animator>(draw_frame, draw_frame+30,
                                              &point->location,parent_loc,
                                              loc);
            world.add_object(point,1);
        }
        draw_frame += ceil(30/(i+1));
    }

    // Adding elements to world
    for (int i = 0; i < 4; ++i){
        world.add_object(square[i], 2);
    }
    for (int i = 0; i < 5; ++i){
        world.add_object(midsquare[i], 2);
    }

    for (int i = 0; i < total_frames; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();

}

void sierpinski_hutchinson(camera& cam, scene& world, int level,
                           int total_frames, bool aaa_version){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color blue  = {0,0,1,1};
    color red   = {1,0,0,1};
    color green = {0,1,0,1};
    color gray  = {0.5,0.5,0.5,1};

    color pt_clr, bg_clr;
    if (aaa_version){
        pt_clr = black;
        bg_clr = white;
    }
    else{
        pt_clr = white;
        bg_clr = black;
    }

    double text_size = 100;
    double ball_size = (level*2);

    std::vector<vec> triangle_pts = {{0,0},{0.5,sqrt(0.75)},{1,0}};
    std::vector<vec> triangle_midpts = {0.5*(triangle_pts[0]+triangle_pts[1]),
                                        0.5*(triangle_pts[1]+triangle_pts[2]),
                                        0.5*(triangle_pts[2]+triangle_pts[0])};
    
    std::vector<std::shared_ptr<ellipse>> triangle(3), midtriangle(3);

    int triangle_size = world.size.y*0.95;
    int triangle_offset = triangle_size*0.5;
    std::shared_ptr<text> label;
    vec label_offset;

    std::string letter;
    // Three triangle points
    for (int i = 0; i < triangle.size(); ++i){
        switch(i){
            case 0:
                letter = "A";
                label_offset = {0, text_size};
                break;
            case 1:
                letter = "B";
                label_offset = {0, -0.25*text_size};
                break;
            case 2:
                letter = "C";
                label_offset = {0, text_size};
                break;
        }
        vec loc = {world.size.x*0.5 - triangle_offset 
                   + triangle_size*triangle_pts[i].x,
                   world.size.y*0.5 + triangle_offset*sqrt(0.75) 
                   - triangle_size*triangle_pts[i].y};
        triangle[i] = std::make_shared<ellipse>(pt_clr, loc, vec{0,0}, 0, 1);
        triangle[i]->add_animator<vec_animator>(0+i*10,30+i*10,
                                                &triangle[i]->size,
                                                vec{0,0},
                                                vec{ball_size, ball_size});

        label = std::make_shared<text>(bg_clr, loc+label_offset, text_size,
                                       letter, 0);
        label->add_animator<color_animator>(0+i*10,30+i*10,
                                           &label->clr,
                                           bg_clr,pt_clr);
        world.add_object(label, 0);

        color label_clr;
        switch(i){
            case 0:
                letter = "D";
                label_offset = {-1*text_size,0};
                label_clr = red;
                break;
            case 1:
                letter = "E";
                label_offset = {0.25*text_size, 0};
                label_clr = green;
                break;
            case 2:
                letter = "F";
                label_offset = {0, text_size};
                label_clr = blue;
                break;
        }
        loc = {world.size.x*0.5 - triangle_offset 
               + triangle_size*triangle_midpts[i].x,
               world.size.y*0.5 + triangle_offset*sqrt(0.75) 
               - triangle_size*triangle_midpts[i].y};
        midtriangle[i] = std::make_shared<ellipse>(pt_clr, loc, vec{0,0}, 0, 1);
        midtriangle[i]->add_animator<vec_animator>(60+i*10,90+i*10,
                                                &midtriangle[i]->size,
                                                vec{0,0},
                                                vec{ball_size, ball_size});

        label = std::make_shared<text>(bg_clr, loc+label_offset, text_size,
                                       letter, 0);
        label->add_animator<color_animator>(60+i*10,90+i*10,
                                           &label->clr,
                                           bg_clr,pt_clr);
        label->add_animator<color_animator>(120+i*40,150+i*40,
                                           &label->clr,
                                           pt_clr, label_clr);
        midtriangle[i]->add_animator<color_animator>(120+i*40,150+i*40,
                                                     &midtriangle[i]->clr,
                                                     pt_clr, label_clr);
        world.add_object(label, 0);
    }

    vec loc, parent_loc;
    auto point = std::make_shared<ellipse>(pt_clr, vec{0,0}, vec{10,10}, 0, 1);

    world.add_layer();
    world.add_layer();

    tryte value (0,0);
    int tmp_level = 1;
    int diff = 3;
    std::string value_string = "", parent_string;

    int draw_frame = 250;

    // This is creating all the children
    for(int i = 0; i < ((pow(3,level)-1)/2)-1; i++){
        if (i == diff){
            tmp_level += 1;
            value = {0,0};
            diff += pow(3,tmp_level);
            draw_frame += 60;
            if (ball_size > 1){
                ball_size /= 1.4;
            }
            else{
                ball_size = 1;
            }
        }
        value = increment(value);
        value_string = save_tryte(value,tmp_level);
        for (int j = 0; j < 3; ++j){
            color point_clr;
            switch(j){
                case 0:
                    point_clr = red;
                    break;
                case 1:
                    point_clr = green;
                    break;
                case 2:
                    point_clr = blue;
                    break;
            }
            loc = apply_tryte(value_string, triangle_midpts[j],
                              triangle_pts);
            parent_string = value_string.substr(0,value_string.size()-1);
            parent_loc = apply_tryte(parent_string, triangle_midpts[j],
                                     triangle_pts);
            loc = {world.size.x*0.5 - triangle_offset + triangle_size*loc.x,
                   world.size.y*0.5 + triangle_offset*sqrt(0.75)
                   - triangle_size*loc.y};
            parent_loc = {world.size.x*0.5 - triangle_offset
                          + triangle_size*parent_loc.x,
                          world.size.y*0.5 + triangle_offset*sqrt(0.75)
                          - triangle_size*parent_loc.y};
            point = std::make_shared<ellipse>(point_clr, loc,
                                              vec{0,0}, 0, 1);
            point->add_animator<vec_animator>(draw_frame, draw_frame+30,
                                              &point->size,vec{0,0},
                                              vec{ball_size, ball_size});
            point->add_animator<vec_animator>(draw_frame, draw_frame+30,
                                              &point->location,parent_loc,
                                              loc);
            world.add_object(point,1);
        }
        draw_frame += ceil(30/(i+1));
    }

    // Adding elements to world
    for (int i = 0; i < 3; ++i){
        world.add_object(triangle[i], 2);
        world.add_object(midtriangle[i], 2);
    }

    for (int i = 0; i < total_frames; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();

}

void sierpinski_chaos(camera& cam, scene& world, int n, int bin_size,
                      int total_frames, bool aaa_version){
    color black = {0,0,0,1};
    color white = {1,1,1,1};
    color gray = {0.5,0.5,0.5,1};

    color pt_clr, bg_clr;
    if (aaa_version){
        pt_clr = black;
        bg_clr = white;
    }
    else{
        pt_clr = white;
        bg_clr = black;
    }

    double text_size = 100;
    double ball_size = (14);

    std::vector<vec> triangle_pts = {{0,0},{0.5,sqrt(0.75)},{1,0}};
    std::vector<std::shared_ptr<ellipse>> triangle(3);

    int triangle_size = world.size.y*0.95;
    int triangle_offset = triangle_size*0.5;

    std::shared_ptr<text> label;
    std::string letter;
    vec label_offset;

    // Three triangle points
    for (int i = 0; i < triangle.size(); ++i){
        switch(i){
            case 0:
                letter = "A";
                label_offset = {0, text_size};
                break;
            case 1:
                letter = "B";
                label_offset = {0, -0.25*text_size};
                break;
            case 2:
                letter = "C";
                label_offset = {0, text_size};
                break;
        }

        vec loc = {world.size.x*0.5 - triangle_offset
                   + triangle_size*triangle_pts[i].x,
                   world.size.y*0.5 + triangle_offset*sqrt(0.75)
                   - triangle_size*triangle_pts[i].y};
        triangle[i] = std::make_shared<ellipse>(pt_clr, loc, vec{0,0}, 0, 1);
        triangle[i]->add_animator<vec_animator>(0+i*10,30+i*10,
                                                &triangle[i]->size,
                                                vec{0,0},
                                                vec{ball_size, ball_size});

        label = std::make_shared<text>(bg_clr, loc+label_offset, text_size,
                                       letter, 0);
        label->add_animator<color_animator>(0+i*10,30+i*10,
                                           &label->clr,
                                           bg_clr,pt_clr);
        world.add_object(label, 0);
    }

    // Adding elements to world
    world.add_layer();
    world.add_layer();
    world.add_layer();
    for (int i = 0; i < 3; ++i){
        world.add_object(triangle[i], 2);
    }


    // First, generate random point.
    srand(1337);
    //vec pt = {rand() % 10000 * 0.0001, rand() % 10000 * 0.0001};
    vec pt = {0.5, 0.25}, pts[20];
    vec prev_pt = {0.5, 0.25};

    //std::vector<std::share_ptr<ellipse>> balls;
    // Implementing Sierpinski triangle
    for (int i = 0; i < n; ++i){
        vec prev_loc = {world.size.x*0.5 - triangle_offset 
                        + triangle_size*prev_pt.x,
                        world.size.y*0.5 + triangle_offset*sqrt(0.75)
                        - triangle_size*prev_pt.y};
        vec loc = {world.size.x*0.5 - triangle_offset + triangle_size*pt.x,
                   world.size.y*0.5 + triangle_offset*sqrt(0.75)
                   - triangle_size*pt.y};

        if (i < 20){
            double ball_size = 14;
            color ball_color = {0, 1.0-i/20.0, i/20.0, 1};
            auto ball = std::make_shared<ellipse>(ball_color,
                                                  loc, vec{0,0}, 0, 1);
            ball->add_animator<vec_animator>(60+i*30+floor(n/bin_size)-20,
                                             90+i*30+floor(n/bin_size)-20,
                                             &ball->size,
                                             vec{0,0},
                                             vec{ball_size, ball_size});

            color clear = {0,0,0,0};
            color line_color;
            if (aaa_version){
                line_color = {0, 0, 0, 1.0-(i+1)/20.0};
            }
            else{
                line_color = {1, 1, 1, 1.0-(i+1)/20.0};
            }
            auto arrow = std::make_shared<line>(black, prev_loc, prev_loc);
            arrow->add_animator<color_animator>(60+i*30+floor(n/bin_size)-20,
                                                61+i*30+floor(n/bin_size)-20,
                                                &arrow->clr, line_color,
                                                line_color);
            arrow->add_animator<vec_animator>(60+i*30+floor(n/bin_size)-20,
                                              90+i*30+floor(n/bin_size)-20,
                                              &arrow->end, prev_loc, loc);
            world.add_object(ball,3);
            world.add_object(arrow,2);
        }
        else{
            double ball_size = 2;
            color ball_color = {1-(double)i/n, 0, 1-((double)n-i)/n,1};
            auto ball = std::make_shared<ellipse>(ball_color,
                                                  loc, vec{0,0}, 0, 1);
            ball->add_animator<vec_animator>(30+floor(i/bin_size)-20,
                                             60+floor(i/bin_size)-20,
                                             &ball->size,
                                             vec{0,0},
                                             vec{ball_size, ball_size});
            world.add_object(ball,1);
        }

        prev_pt = pt;
        double rnd = rand() % 10000 * 0.0001;
        if (rnd <= 0.33){
            pt = midpoint(triangle_pts[0],pt);
        }
        else if (rnd > 0.33 && rnd <= 0.66){
            pt = midpoint(triangle_pts[1],pt);
        }
        else{
            pt = midpoint(triangle_pts[2],pt);
        }

    }


    for (int i = 0; i < total_frames; ++i){
        world.update(i);
        cam.encode_frame(world);
    }

    world.clear();
}

int main(){

    print_increment();
/*
    camera cam(vec{2*1920, 2*1080});
    scene world = scene({2*1920, 2*1080}, {0, 0, 0, 1});
    //camera cam(vec{1920, 1080});
    //scene world = scene({1920, 1080}, {0, 0, 0, 1});
    //world.bg_clr = {1, 1, 1, 1};

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);
    //cam.add_encoder<png_encoder>();
    //square_hutchinson(cam, world, 7, 1000, false);
    sierpinski_hutchinson(cam, world, 8, 1000, false);
    //sierpinski_chaos(cam, world, 200000, 1000, 1000, false);
    cam.clear_encoders();
*/
}
