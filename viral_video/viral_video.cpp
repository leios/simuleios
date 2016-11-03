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

    // Defining necessary information for visualization
    frame anim;
    anim.create_frame(400,300,30,"/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0, 0, 0);

    std::vector<viewer> viewers;
    int num_videos = 1000;
    viewers = init_viewers(anim, 100, num_videos, 10);

    // Creating file for outputting curr_video later.
    std::ofstream output("out.dat", std::ofstream::out);

/*
    for (int i = 0; i < viewers.size(); i++){
        std::cout << i << '\t' << viewers[i].videos_seen[0] << '\t' 
                  << viewers[i].affinities[0] << '\n';
    }
*/

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

    output.close();
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
    int j = 0;
    for (int i = 0; i < viewer_num; i++){
        viewers[i].p.x = (anim.res_x / box_size) * (i - j*floor(box_size));
        viewers[i].p.y = (anim.res_y / box_size) * j;
        if (i == floor(box_size)){
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
                network.reserve(network_dist(gen));
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
        viewers[i].next_video = network[i];
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
    for (int i = 0; i < viewers.size(); i++){
        draw_human(anim, viewers[i].p, 10, white);
    }    
}

