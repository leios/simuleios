/*-------------viral_video.cpp------------------------------------------------//
*
* Purpose: To simulate how a video becomes viral
*
*   Notes: videos represented as integers, we don't care about their name
*          videos_seen will hold history for the viewer, we may cut this off
*          viewers have seen ~10 videos at the start of the simulation
*          the video we are propagating is video 0
*          Now, viewers may watch the same videos over gain, check affinity
*          Compile with "g++ viral_video.cpp -std=c++11"
*
*          25% chance to share if they liked the video
*          Don't forget to call "find_affinities" after watch_videos
*
*          the -fsanatize=address faile at creating random device rd
*
*-----------------------------------------------------------------------------*/

#include "viral_video.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    //visualize_large_network();
    visualize_moving_network(100);
    //visualize_small_network(100);
    //visualize_pull(10);

/*
    // Defining necessary information for visualization
    frame anim;
    anim.create_frame(400,300,30,"/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0, 0, 0);
    color white = {1,1,1,1};
    vec human_pos = {anim.res_x * 0.5, anim.res_y * 0.5};
    vec text_pos = {anim.res_x * 0.5, anim.res_y * 0.2};
    animate_human(anim, human_pos, anim.res_y * 0.5, white, 3);
    std::cout << anim.curr_frame << '\n';
    write_text(anim, human_pos, text_pos, anim.res_y * 0.1616, 25, "hey");

    std::vector<viewer> viewers;
    int num_videos = 1000;
    viewers = init_viewers(anim, 100, num_videos, 10);

    // Creating file for outputting curr_video later.
    std::ofstream output("out.dat", std::ofstream::out);

    provide_videos(viewers, 10, num_videos);

    for (int i = 0; i < 100; i++){
        provide_videos(viewers, 10, num_videos);
        choose_video(viewers);
        output_viewers(viewers, output);
        watch_video(viewers);
        for (int j = 0; j < viewers.size(); j++){
            find_affinities(j, viewers);
        }
    }

    animate_viewers(anim, viewers);
    for (int i = anim.curr_frame; i < 300; i++){
        draw_viewers(anim, viewers);
        anim.curr_frame++;
    }
    anim.draw_frames();

    //output.close();
*/
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(frame &anim, int viewer_num, int num_videos, 
                                 int subscribers){

    int history = 10;
    video chosen_video;

    // if sqrt(viewer_num) != int, make squarable
    viewer_num = ceil(sqrt(viewer_num)*sqrt(viewer_num));

    // Defining random distribution for videos seen by people
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        chosen_videos_dist(1, num_videos-1);

    std::uniform_int_distribution<int>
        viewer_dist(0, viewer_num-1);

    std::vector<viewer> viewers(viewer_num);

    // Defining videos seen by each person
    for (int i = 0; i < viewer_num; i++){
        //viewers[i].videos_seen.reserve(history);
        viewers[i].video_list.reserve(10);
        viewers[i].index = i;
        for (int j = 0; j < history; j++){
            chosen_video.id = chosen_videos_dist(gen);
            viewers[i].videos_seen[chosen_video.id] = 1.0;
        }
    }

    // Defining all the initial affinities
    for (int i = 0; i < viewer_num; i++){
        viewers[i].affinities = find_affinities(i, viewers);
    }

    // Selecting certain people to watch video 0 at the start of the simulation
    int sub;
    for (int i = 0; i < subscribers; i++){
        sub = viewer_dist(gen);   
        viewers[sub].next_video = 0;
    }

    // Providing all people an initial position
    // Checking to see if we can easily put everyone in a square
    double box_size = sqrt((double)viewer_num);
    std::cout << box_size << '\n';
    int j = 0;
    int k = -1;
    double offset_x = anim.res_x * 0.1;
    double offset_y = anim.res_y * 0.1;
    for (int i = 0; i < viewer_num; i++){
        k++;
        viewers[i].p.x = 0.5 * offset_x + ((anim.res_x - offset_x) / box_size) 
                                           * (k);
        viewers[i].p.y = 0.5 * offset_y + ((anim.res_y-offset_y)/box_size)*j;
        viewers[i].pp = viewers[i].p;
        viewers[i].vel.x = 0;
        viewers[i].vel.y = 0;
        viewers[i].acc.x = 0;
        viewers[i].acc.y = 0;
 
        if (k == box_size - 1){
            k = -1;
            j++;
        }
    }

    return viewers;
}

// Function to find affinity with all other people 
std::vector<double> find_affinities(int individual,
                                    std::vector<viewer> &viewers){

    std::vector<double> affinities(viewers.size());
    std::vector<video>::iterator it;
    double affinity;
    int shared;

    // Searching through all the other viewers to find videos in common
    for (int i = 0; i < viewers.size(); i++){
        shared = 0;
        if (individual != i){
            // Naive method to find all videos shared between to folks
            for (const auto& vid : viewers[individual].videos_seen){
                if (viewers[i].videos_seen.find(vid.first) != 
                    viewers[i].videos_seen.end()){
                    shared++;
                }
            }
/*
            for (int j = 0; j < viewers[individual].videos_seen.size(); j++){
                it = find(viewers[i].videos_seen.begin(), 
                          viewers[i].videos_seen.end(),
                          viewers[individual].videos_seen[j]);
                if (it != viewers[i].videos_seen.end()){
                    shared++;
                }
            }
*/
            affinity = (double)shared 
                       / (double)viewers[individual].videos_seen.size();
            affinities[i] = affinity;

            shared = 0;
        }
        else{
            affinities[i] = 0;
        }
    }

    return affinities;
}

// Function to provide videos to all viewers
void provide_videos(std::vector<viewer> &viewers, int cutoff, int num_videos){

    // Creating list for how individuals see the rest of the community.
    std::vector<viewer> similarity_list = viewers;
    std::vector<video> video_values(num_videos);

    for (int i = 0; i < video_values.size(); i++){
        video_values[i].id = i;
    }

    // Updating video list for all viewers
    for (int i = 0; i < viewers.size(); i++){

        // Sorting procedure written by Gustorn
        std::sort(similarity_list.begin(), similarity_list.end(),
                  [&viewers, &i](viewer& a, viewer& b){
                      return viewers[i].affinities[a.index] >
                             viewers[i].affinities[b.index];
                  });

        // Populating the video_values with doubles 
        for (int j = 0; j < cutoff; j++){
            // Going through people's video history
            for (auto &k : similarity_list[j].videos_seen){
                //std::cout << k.first << '\n';
                video_values[k.first].val +=  
                    viewers[i].affinities[similarity_list[j].index]
                    * k.second;
            }
        }

        std::sort(video_values.begin(), video_values.end(), 
                  [](const video& a, const video& b){return a.val > b.val;});

/*
        for (int i = 0; i < video_values.size(); i++){
            std::cout << video_values[i].id << '\t' << video_values[i].val 
                      << '\n';
        }
*/

        viewers[i].video_list.clear();
        for (int j = 0; j < 10; j++){
            viewers[i].video_list.push_back(video_values[j]);
        }

        // Setting the similarity list back to what it was before.
        similarity_list = viewers;
    }

/*
    for (int i = 0; i < viewers.size(); i++){
        std::cout << viewers[i].video_list[5].id << '\n';
    }
*/
}

// Function for each individual to choose a video from their video_list and 
// rate it randomly between 0, 1, and 2
void choose_video(std::vector<viewer> &viewers){

    // Defining a second distribution for the choice of videos
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        choice_dist(0, viewers[0].video_list.size() - 1);

    // Going through our list of individuals and forcing them to watch videos
    for (int i = 0; i < viewers.size(); i++){
        if (viewers[i].next_video < 0){
            viewers[i].curr_video = viewers[i].video_list[choice_dist(gen)].id;
        }
        else{
            viewers[i].curr_video = viewers[i].next_video;
            viewers[i].next_video = -1;
        }
        if (viewers[i].curr_video == 0){
            viewers[i].has_seen_0 = 1;
        }
    }
}

// Function for each individual to choose a video from their video_list and 
// rate it randomly between 0, 1, and 2
void watch_video(std::vector<viewer> &viewers){

    // Defining random distribution for how much people like vides
    // Cast to double when storing.
    // Three possible values for video feedback: 0 -- dislike
    //                                           1 -- nothing
    //                                           2 -- like
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        like_dist(0, 2);

    // Distribution for setting each individual's network
    std::uniform_int_distribution<int>
        network_dist(0, viewers.size()-1);
    std::uniform_int_distribution<int>
        network_size_dist(0, (viewers.size()-1) / 10);

    // Distribution for random chance to share
    std::uniform_real_distribution<double>
        share_dist(0,1);

    int video_choice, like_choice;
    double share_chance;

    // Going through our list of individuals and forcing them to watch videos
    // They then tell us how much they like the video.
    for (int i = 0; i < viewers.size(); i++){
        video_choice = viewers[i].curr_video;
        like_choice = like_dist(gen);
        viewers[i].curr_video = -1;
 
        viewers[i].videos_seen[video_choice] = (double)like_choice;
        if (like_choice > 1){
            share_chance = share_dist(gen);
            if (share_chance > 0.50){

                // Creating this person's network
                std::vector <int> network;
                network.reserve(network_size_dist(gen));
                for (int i = 0; i < network.size(); i++){
                    network.push_back(network_dist(gen));
                }
                share(viewers, network, video_choice);
            }
        }
    }
}

// Function to share videos with other people randomly in the network
// Note that this simply forces the individuals to watch a particular video 
// next timestep
void share(std::vector<viewer> &viewers, std::vector<int> &network, 
           int video_id){
    for (int i = 0; i < network.size(); i++){
        viewers[network[i]].next_video = video_id;
    }
}

// Function to output which videos are being watched to a file.
void output_viewers(std::vector<viewer> &viewers, std::ofstream &file){

    for (int i = 0; i < viewers.size(); i++){
        file << i << '\t' << viewers[i].has_seen_0 << '\n';
        //file << i << '\t' << viewers[i].curr_video << '\n';
    }

    file << '\n' << '\n';
}

// Function to place viewers in a frame for visualization
void draw_viewers(frame &anim, std::vector<viewer> &viewers){
    color white = {1,1,1,1};
    color red = {1, 0, 0, 1};
    color viewer_clr = white;
    for (int i = 0; i < viewers.size(); i++){
        if (viewers[i].has_seen_0 == 1){
            viewer_clr = red;
        }
        else{
            viewer_clr = white;
        }
        draw_human(anim, viewers[i].p, 10, viewer_clr);
    }    
}

// Function to draw viewers as pixels in an image
void draw_pixel_viewers(frame &anim, std::vector<viewer> &viewers){
    color black = {0,0,0,1};
    color red = {1, 0, 0, 1};
    color viewer_clr = black;

    int x_pos = 0;
    int y_pos = 0;

    for (int i = 0; i < anim.res_x; i++){
        for (int j = 0; j < anim.res_y; j++){
            if (viewers[i].has_seen_0 == 1){
                viewer_clr = red;
            }
            else{
                viewer_clr = black;
            }
            x_pos = i;
            y_pos = j;
            std::cout << x_pos << '\t' << y_pos << '\n';
            cairo_rectangle(anim.frame_ctx[anim.curr_frame], x_pos, y_pos,1,1);
            cairo_set_source_rgba(anim.frame_ctx[anim.curr_frame], viewer_clr.r,
                                  viewer_clr.g, viewer_clr.b, viewer_clr.a);
            cairo_fill(anim.frame_ctx[anim.curr_frame]);
        }

    }

    std::cout << '\n' << '\n';
}

// Function to draw all viewers successively
void animate_viewers(frame &anim, std::vector<viewer> &viewers){
    color white = {1,1,1,1};
    for (int i = 0; i < viewers.size(); i++){
        for (int j = 0; j < anim.curr_frame; j++){
            draw_human(anim, viewers[j].p, 10, white);
        }
        anim.curr_frame++;
    }
}

// Function to visualize a pull network
void visualize_pull(int viewer_num){

    color white = {1,1,1,1};

    // Two layers, one for people, zero for lines
    std::vector<frame> layers(2);
    for (int i = 0; i < layers.size(); i++){
        layers[i].create_frame(400,300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0],0,0,0);
    double height = 0.1 * layers[0].res_y;

    // I suppose a random distribution of people should work.
    // Note the dist is from 0-1 and will be multiplied by res_x,y later
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        pos_dist(0, 1);

    // Now we need a vector for all the human positions
    // The first position should be at the center of the screen
    std::vector<vec> positions(viewer_num+1);
    positions[0].x = 0.5 * layers[0].res_x;
    positions[0].y = 0.5 * layers[0].res_y;

    for (int i = 1; i < positions.size(); i++){
        positions[i].x = pos_dist(gen) * layers[0].res_x;
        positions[i].y = pos_dist(gen) * layers[0].res_y;
    }

    // now we need to draw and connect lines to all of them
    // We need to draw all the humans on all the frames...
    // TODO
    int curr_frame = layers[1].curr_frame;
    int final_count;
    for (int i = layers[1].curr_frame; i < num_frames; i++){
        if (i < positions.size()){
            final_count = i;
        }
        else{
            final_count = positions.size();
        }
        for (int j = 0; j < final_count; j++){
            draw_human(layers[1], positions[j], height, white);
        }
        layers[1].curr_frame++;
    }

    layers[0].curr_frame = curr_frame + positions.size();
    layers[1].curr_frame = curr_frame + positions.size();

    curr_frame = layers[0].curr_frame;

    // We will need to draw the lines to each respective head... but we don't
    // want to overlap with the skull, so we need to find the angle and such
    // and then add the head radius for the temp positions
    for (int i = 1; i < positions.size(); i++){
        animate_line(layers[0], curr_frame, 1, positions[i],
                     positions[0], white);
    }

    // Now we need to draw orbs that extend from the center person to all
    // People on the outside
    // draw_frames is going to be the number of frames we are drawing on for 2s
    int draw_frames = layers[0].fps * 2.0;
    int index = 0;
    vec temp_dist;
    double x_pos, y_pos;

    // Going from random group to central individual
    for (int i = 1; i < positions.size(); i++){
        temp_dist.x = (positions[i].x - positions[0].x);
        temp_dist.y = (positions[i].y - positions[0].y);
        for (int j = 0; j < draw_frames; j++){
            index = layers[0].curr_frame + j;
            x_pos = positions[i].x - j * (temp_dist.x / draw_frames);
            y_pos = positions[i].y - j * (temp_dist.y / draw_frames);
            cairo_arc(layers[0].frame_ctx[index],x_pos, y_pos, height * 0.33, 
                      0,2*M_PI);
            cairo_fill(layers[0].frame_ctx[index]);
        }
    }

    // Updating frames
    layers[0].curr_frame += draw_frames;

    // Going back
    for (int i = 1; i < positions.size(); i++){
        temp_dist.x = (positions[0].x - positions[i].x);
        temp_dist.y = (positions[0].y - positions[i].y);
        for (int j = 0; j < draw_frames; j++){
            index = layers[0].curr_frame + j;
            x_pos = positions[0].x - j * (temp_dist.x / draw_frames);
            y_pos = positions[0].y - j * (temp_dist.y / draw_frames);
            cairo_arc(layers[0].frame_ctx[index],x_pos, y_pos, height * 0.33, 
                      0,2*M_PI);
            cairo_fill(layers[0].frame_ctx[index]);
        }
    }

    draw_layers(layers);

}

// Function to visualize a push network
void visualize_push(int viewer_num){

    color white = {1,1,1,1};

    // Two layers, one for people, zero for lines
    std::vector<frame> layers(2);
    for (int i = 0; i < layers.size(); i++){
        layers[i].create_frame(400,300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0],0,0,0);
    double height = 0.1 * layers[0].res_y;

    // I suppose a random distribution of people should work.
    // Note the dist is from 0-1 and will be multiplied by res_x,y later
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        pos_dist(0, 1);

    // Now we need a vector for all the human positions
    // The first position should be at the center of the screen
    std::vector<vec> positions(viewer_num+1);
    positions[0].x = 0.5 * layers[0].res_x;
    positions[0].y = 0.5 * layers[0].res_y;

    for (int i = 1; i < positions.size(); i++){
        positions[i].x = pos_dist(gen) * layers[0].res_x;
        positions[i].y = pos_dist(gen) * layers[0].res_y;
    }

    // now we need to draw and connect lines to all of them
    // We need to draw all the humans on all the frames...
    // TODO
    int curr_frame = layers[1].curr_frame;
    int final_count;
    for (int i = layers[1].curr_frame; i < num_frames; i++){
        if (i < positions.size()){
            final_count = i;
        }
        else{
            final_count = positions.size();
        }
        for (int j = 0; j < final_count; j++){
            draw_human(layers[1], positions[j], height, white);
        }
        layers[1].curr_frame++;
    }

    layers[0].curr_frame = curr_frame + positions.size();
    layers[1].curr_frame = curr_frame + positions.size();

    curr_frame = layers[0].curr_frame;

    // We will need to draw the lines to each respective head... but we don't
    // want to overlap with the skull, so we need to find the angle and such
    // and then add the head radius for the temp positions
    for (int i = 1; i < positions.size(); i++){
        animate_line(layers[0], curr_frame, 1, positions[0],
                     positions[i], white);
    }

    // Now we need to draw orbs that extend from the center person to all
    // People on the outside
    // draw_frames is going to be the number of frames we are drawing on for 2s
    int draw_frames = layers[0].fps * 2.0;
    int index = 0;
    vec temp_dist;
    double x_pos, y_pos;
    for (int i = 1; i < positions.size(); i++){
        temp_dist.x = (positions[0].x - positions[i].x);
        temp_dist.y = (positions[0].y - positions[i].y);
        for (int j = 0; j < draw_frames; j++){
            index = layers[0].curr_frame + j;
            x_pos = positions[0].x - j * (temp_dist.x / draw_frames);
            y_pos = positions[0].y - j * (temp_dist.y / draw_frames);
            cairo_arc(layers[0].frame_ctx[index],x_pos, y_pos, height * 0.33, 
                      0,2*M_PI);
            cairo_fill(layers[0].frame_ctx[index]);
        }
    }

    //layers[0].draw_frames();
    draw_layers(layers);

}

// Function to visualize a small network of viewers
void visualize_small_network(int viewer_num){

    // Defining necessary information for visualization
    frame anim;
    anim.create_frame(400,300,30,"/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0, 0, 0);

    std::vector<viewer> viewers;
    int num_videos = 1000;
    viewers = init_viewers(anim, viewer_num, num_videos, 10);

    // Creating file for outputting curr_video later.
    std::ofstream output("out.dat", std::ofstream::out);

    provide_videos(viewers, 10, num_videos);

    animate_viewers(anim, viewers);

    for (int i = anim.curr_frame; i < num_frames; i++){
        provide_videos(viewers, 10, num_videos);
        choose_video(viewers);
        watch_video(viewers);
        draw_viewers(anim, viewers);
        for (int j = 0; j < viewers.size(); j++){
            find_affinities(j, viewers);
        }
        anim.curr_frame++;
    }

    anim.draw_frames();

}

// Function to visualize lage network -- size of canvas
void visualize_large_network(){

    int res_x = 16;
    int res_y = 16;

    // Defining necessary information for visualization
    frame anim;
    anim.create_frame(res_x,res_y,30,"/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0, 0, 0);

    int viewer_num = res_x * res_y;
    std::vector<viewer> viewers;
    int num_videos = 1000;
    viewers = init_viewers(anim, viewer_num, num_videos, 10);


    // Creating file for outputting curr_video later.
    std::ofstream output("out.dat", std::ofstream::out);

    provide_videos(viewers, 10, num_videos);

    for (int i = anim.curr_frame; i < num_frames; i++){
    //for (int i = anim.curr_frame; i < 102; i++){
        choose_video(viewers);
        watch_video(viewers);
        draw_pixel_viewers(anim, viewers);
        for (int j = 0; j < viewers.size(); j++){
            find_affinities(j, viewers);
        }
        anim.curr_frame++;
    }

    anim.draw_frames();

    
}

// Function to visualize a small, moving network of viewers
void visualize_moving_network(int viewer_num){

    // Defining necessary information for visualization
    frame anim;
    anim.create_frame(400,300,30,"/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0, 0, 0);

    std::vector<viewer> viewers;
    int num_videos = 1000;
    viewers = init_viewers(anim, viewer_num, num_videos, 10);

    // Creating file for outputting curr_video later.
    std::ofstream output("out.dat", std::ofstream::out);

    provide_videos(viewers, 10, num_videos);

    //animate_viewers(anim, viewers);

    for (int i = anim.curr_frame; i < num_frames; i++){
    //for (int i = anim.curr_frame; i < 102; i++){
        update_pos(viewers, 1);
        provide_videos(viewers, 10, num_videos);
        choose_video(viewers);
        watch_video(viewers);
        draw_viewers(anim, viewers);
        for (int j = 0; j < viewers.size(); j++){
            find_affinities(j, viewers);
        }
        anim.curr_frame++;
    }

    anim.draw_frames();

}

// Function to find all the accelerations for all the folks
void find_acc(int individual, std::vector<viewer> &viewers){

    double damp = .0001;
    double cutoff = 1.0;
    vec dist;
    viewers[individual].acc.x = 0;
    viewers[individual].acc.y = 0;
    for (int i = 0; i < viewers.size(); i++){
        if (i != individual){
            dist.x = viewers[i].p.x - viewers[individual].p.x;
            dist.y = viewers[i].p.y - viewers[individual].p.y;

            if (viewers[individual].affinities[i] > 0 && 
                abs(dist.x) > cutoff && abs(dist.y) > cutoff){
                viewers[individual].acc.x += sign(dist.x) * 0.0000001
                                            * viewers[individual].affinities[i] 
                                            * (dist.x*dist.x) 
                                            - damp * viewers[individual].vel.x;
                viewers[individual].acc.y += sign(dist.y) * 0.0000001
                                             * viewers[individual].affinities[i]
                                            * (dist.y*dist.y) 
                                            - damp * viewers[individual].vel.y;
            }

        }
    }
}

// Function for the verlet agorithm / position change
void verlet(viewer &individual, double dt){
    vec temp = individual.p;
    individual.p.x = 2 * individual.p.x - individual.pp.x 
                     + individual.acc.x * dt * dt;
    individual.p.y = 2 * individual.p.y - individual.pp.y 
                     + individual.acc.y * dt * dt;
    individual.pp = temp;

    individual.vel.x = (individual.p.x - individual.pp.x) / (2*dt);
    individual.vel.y = (individual.p.y - individual.pp.y) / (2*dt);
}

// Function to update positions every timestep
void update_pos(std::vector<viewer> &viewers, double dt){
    for (int i = 0; i < viewers.size(); i++){
        find_acc(i, viewers);
        verlet(viewers[i], dt);
        //std::cout << viewers[i].acc.x << '\t' << viewers[i].acc.y << '\n';
    }
}

// Return the sign of a double
double sign(double x){
    if (x < 0){
        //std::cout << -1 << '\n';
        return -1;
    }
    else if (x == 0){
        return 0;
    }
    else {
        //std::cout << x << '\t' << 1 << '\n';
        return 1;
    }
}
