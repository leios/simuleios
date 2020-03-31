/*------------gauss_vis.cpp----------------------------------------------------/
*
* Purpose: Simple visualization script for difference scenes for computus video
*          2020
*
*   Notes: Compilation instructions:
*              g++ -L/path/to/GathVL -I/path/to/GathVL -lgathvl -o gauss_vis gauss_vis.cpp `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil` -Wl,-rpath,/path/to/GathVL
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>

#include "include/camera.h"
#include "include/scene.h"

void orbit_scene(camera& cam, scene& world, double end_frame, double timestep){

    color sun_clr = {1, 1, 0.5, 1};
    color earth_clr = {0, 1, 1, 1};
    color moon_clr = {0.5, 0.5, 0.5, 1};

    double earth_radius = 200;
    double moon_radius = 100;

    vec earth_offset = vec{sqrt(earth_radius*earth_radius/2),
                           sqrt(earth_radius*earth_radius/2)};

    vec moon_offset = vec{sqrt(moon_radius*moon_radius/2),
                          sqrt(moon_radius*moon_radius/2)};


    vec sun_loc = {world.size.x/2, world.size.y/2};
    vec earth_loc = sun_loc + earth_offset;
    vec moon_loc = earth_loc + moon_offset;

    auto sun = std::make_shared<ellipse>(sun_clr,
                                         sun_loc,
                                         vec{0,0}, 0, true);

    sun->add_animator<vec_animator>(0,20, &sun->size, vec{0,0}, vec{60,60});
    sun->add_animator<vec_animator>(20,25, &sun->size, vec{60,60}, vec{55,55});

    std::shared_ptr<ellipse> earths[3];
    std::shared_ptr<ellipse> moons[3];
    for (int i = 0; i < 3; ++i){
        earths[i] = std::make_shared<ellipse>(earth_clr, earth_loc,
                                             vec{0,0}, 0, true);

        earths[i]->add_animator<vec_animator>(10,30, &earths[i]->size,
                                              vec{0,0}, vec{30,30});
        earths[i]->add_animator<vec_animator>(30,35, &earths[i]->size,
                                              vec{30,30}, vec{25,25});

        moons[i] = std::make_shared<ellipse>(moon_clr, moon_loc,
                                             vec{0,0}, 0, true);

        moons[i]->add_animator<vec_animator>(20,40, &moons[i]->size,
                                             vec{0,0}, vec{15,15});
        moons[i]->add_animator<vec_animator>(40,45, &moons[i]->size,
                                             vec{15,15}, vec{10,10});
    }


    // Creating a theta for the earth that changes with time
    double earth_theta[2] = {M_PI/4, 0};
    double moon_theta[2] = {M_PI/4, 0};

    double earth_freq = 0.1;
    double moon_freq = earth_freq*365*12/354;

    // TODO: figure out exact number for lunar cycle
    double gregorian_year_frames = 2*M_PI/earth_freq/timestep;
    double lunar_year_frames = gregorian_year_frames*29.53*12/365.2524;
    int indices = 2;
    for (int i = 50; i < gregorian_year_frames + 51; ++i){
        //if (i > lunar_year_frames + 32){
        if (i == floor(lunar_year_frames) + 51){
            indices = 1;
            earth_theta[1] = fmod(earth_theta[0], 2*M_PI);
        }
        earth_theta[0] -= earth_freq*timestep;
        moon_theta[0] -= moon_freq*timestep + earth_freq*timestep;
        for (int j = 0; j < indices; ++j){
            vec new_loc = sun_loc + vec(earth_radius*cos(earth_theta[0]),
                                        earth_radius*sin(earth_theta[0]));
            earths[j]->add_animator<vec_animator>(i-1,i, &earths[j]->location,
                                                  earth_loc, new_loc);
            earth_loc = new_loc;

            new_loc = earth_loc + vec(moon_radius*cos(moon_theta[0]),
                                      moon_radius*sin(moon_theta[0]));
            moons[j]->add_animator<vec_animator>(i-1,i, &moons[j]->location,
                                                 moon_loc, new_loc);
            moon_loc = new_loc;
        }
    }
    moon_theta[1] = fmod(moon_theta[0], 2*M_PI);
    moon_theta[0] = M_PI/4;

    // Drawing the line for the Earths
    auto earth_arc = std::make_shared<arc>(sun_loc,
                                           vec(earth_radius,earth_radius),
                                           vec{0,0});
    earth_arc->add_animator<vec_animator>(50+gregorian_year_frames,
                                          80+gregorian_year_frames,
                                          &earth_arc->angles,
                                          vec{earth_theta[0], earth_theta[0]},
                                          vec{earth_theta[0], earth_theta[1]});

    // Drawing the line for the Earths
    auto moon_arc = std::make_shared<arc>(earth_loc,
                                          vec(moon_radius,moon_radius),
                                          vec{0,0});
    moon_arc->add_animator<vec_animator>(90+gregorian_year_frames,
                                         120+gregorian_year_frames,
                                         &moon_arc->angles,
                                         vec{moon_theta[1], moon_theta[1]},
                                         vec{moon_theta[1], moon_theta[0]});

    world.add_layer();
    world.add_layer();
    world.add_object(sun, 1);
    for (int i = 0; i < 3; ++i){
        world.add_object(earths[i], 1);
        world.add_object(moons[i], 1);
    }


    for (int i = 0; i < end_frame; ++i){
        world.update(i);
        if (i > 50+gregorian_year_frames){
            world.add_object(earth_arc, 1);
        }
        if (i > 90+gregorian_year_frames){
            world.add_object(moon_arc, 1);
        }
        cam.encode_frame(world);
    }
}

int main() {
    camera cam(vec{1280, 720});
    scene world = scene({1280, 720}, {0, 0, 0, 1});

    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 60);

    orbit_scene(cam, world, 1000, 0.1);

    cam.clear_encoders();

    return 0;
}
