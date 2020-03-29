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

    vec sun_loc = {world.size.x/2, world.size.y/2};
    vec earth_loc = sun_loc + vec{sqrt(earth_radius*earth_radius/2),
                                  sqrt(earth_radius*earth_radius/2)};
    vec moon_loc = earth_loc + vec{sqrt(moon_radius*moon_radius/2),
                                   sqrt(moon_radius*moon_radius/2)};

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
    double earth_theta = M_PI/4;
    double moon_theta = M_PI/4;

    double earth_freq = 0.1;
    double moon_freq = earth_freq*354*12/365;

    // TODO: figure out exact number for lunar cycle
    double gregorian_year_frames = 2*M_PI/earth_freq/timestep;
    double lunar_year_frames = gregorian_year_frames*29.53*12/365.2524;
    int indices = 2;
    for (int i = 50; i < gregorian_year_frames + 51; ++i){
        if (i > lunar_year_frames + 32){
            indices = 1;
        }
        earth_theta -= earth_freq*timestep;
        moon_theta -= moon_freq*timestep;
        for (int j = 0; j < indices; ++j){
            vec new_loc = sun_loc + vec(earth_radius*cos(earth_theta),
                                        earth_radius*sin(earth_theta));
            earths[j]->add_animator<vec_animator>(i-1,i, &earths[j]->location,
                                                  earth_loc, new_loc);
            earth_loc = new_loc;

            new_loc = earth_loc + vec(moon_radius*cos(moon_theta),
                                      moon_radius*sin(moon_theta));
            moons[j]->add_animator<vec_animator>(i-1,i, &moons[j]->location,
                                                 moon_loc, new_loc);
            moon_loc = new_loc;
        }
    }

    world.add_layer();
    world.add_object(sun, 1);
    for (int i = 0; i < 3; ++i){
        world.add_object(earths[i], 1);
        world.add_object(moons[i], 1);
    }

    for (int i = 0; i < end_frame; ++i){
        world.update(i);
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
