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
*          Sort the videos to find the videos to provide.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>

// Struct for videos
struct video{
    int id;
    double val = 1.0;
};

// equality operator for videos
bool operator==(const video& a, const video& b){
    return (a.id == b.id);
}

// Struct to hold all information for each individual viewer
struct viewer{
    std::vector<video> video_list;
    //std::vector<video> videos_seen;
    std::unordered_map<int, double> videos_seen;
    std::vector<double> affinities;
    int index;
};

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(int viewer_num, int num_videos);

// Function to find affinity with all other people 
std::vector<double> find_affinities(int individual,
                                    std::vector<viewer> &viewers);

// Function to provide videos to all viewers
void provide_videos(std::vector<viewer> &viewers, int cutoff, int num_videos);

// Function for each individual to choose a video from their video_list and 
// rate it randomly between 0, 1, and 2
void watch_video(std::vector<viewer> &viewers);

// Function to share videos with other people randomly in the network
// Note that this simply forces the individuals to watch a particular video 
// next timestep
void share(std::vector<viewer> &viewers, std::vector<int> &network, 
           int video_id);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    std::vector<viewer> viewers;
    int num_videos = 10;
    viewers = init_viewers(100, num_videos);

    for (int i = 0; i < viewers.size(); i++){
        std::cout << i << '\t' << viewers[i].videos_seen[0] << '\t' 
                  << viewers[i].affinities[0] << '\n';
    }

    provide_videos(viewers, 10, num_videos);
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(int viewer_num, int num_videos){

    int history = 10;
    video chosen_video;

    // Defining random distribution for videos seen by people
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        chosen_videos_dist(1, num_videos-1);

    std::vector<viewer> viewers(viewer_num);

    // Defining videos seen by each person
    for (int i = 0; i < viewer_num; i++){
        //viewers[i].videos_seen.reserve(history);
        viewers[i].video_list.reserve(10);
        viewers[i].index = i;
        for (int j = 0; j < history; j++){
            chosen_video.id = chosen_videos_dist(gen);
            //viewers[i].videos_seen.push_back(chosen_video);
            viewers[i].videos_seen[chosen_video.id] = 1.0;
        }
    }

    // Denfining all the initial affinities
    for (int i = 0; i < viewer_num; i++){
        std::cout << i << '\n';
        viewers[i].affinities = find_affinities(i, viewers);
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
        std::cout << i << '\n';
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
                video_values[k.first].val +=  
                    viewers[i].affinities[similarity_list[j].index]
                    * k.second;
            }
        }

        std::sort(video_values.begin(), video_values.end(), 
                  [](const video& a, const video& b){return a.val > b.val;});

        viewers[i].video_list.clear();
        for (int j = 0; j < 10; j++){
            viewers[i].video_list.push_back(video_values[j]);
        }

        // Setting the similarity list back to what it was before.
        similarity_list = viewers;
    }

    for (int i = 0; i < viewers.size(); i++){
        std::cout << viewers[i].video_list[5].id << '\n';
    }
}

// Function for each individual to choose a video from their video_list and 
// rate it randomly between 0, 1, and 2
void watch_video(std::vector<viewer> &viewers){
}

// Function to share videos with other people randomly in the network
// Note that this simply forces the individuals to watch a particular video 
// next timestep
void share(std::vector<viewer> &viewers, std::vector<int> &network, 
           int video_id);

