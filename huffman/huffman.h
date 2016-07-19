/*-------------huffman.h------------------------------------------------------//
* 
* Purpose: header file for simple huffman encoding
*
*-----------------------------------------------------------------------------*/

#ifndef HUFFMAN_H
#define HUFFMAN_H

#include<vector>
#include<string>
#include<unordered_map>
#include<queue>

// Struct to hold positions
struct pos{
    double x, y;
};

// Struct for colors
struct color{
    double r, g, b;
};


// Create the binary tree
struct node{
    char key;
    double weight;
    pos ori;

    node() = default;
    node(char k, double w){
        key = k;
        weight = w;
    };
    node *left;
    node *right;
    node *parent;
};

// Struct to compare nodes
struct node_comparer{
    bool operator()(const node *left, const node *right) const {
        return left->weight > right->weight;
    }
};

// Struct to hold bitvalues and also the length of the huffman tree
// (huffman code points)
struct huffman_cp{
    // 32 is arbitrarily set
    //std::bitset<128> code;
    std::string code;
    char key;
};

// Struct to hold final result before decoding
struct huffman_tree{
    node *root;
    std::unordered_map<char, std::string> bitmap;
    std::string phrase, encoded_phrase;
    std::unordered_map<char, double> weightmap;
    int alphabet_size;
};

using node_queue = std::priority_queue<node*,std::vector<node*>,node_comparer>;

// creates nodes from vectors of characters and doubles
node_queue create_nodes(std::vector<char> keys, std::vector<double> weights);
node_queue create_nodes(std::unordered_map<char, double> keyweights);

// Creates the simple binary tree
node* huffman(node_queue &initial_nodes);

// creates bit code
//std::vector<huffman_cp> create_bits(node* root);
std::unordered_map<char, std::string> create_bits(node* root);

// does the encoding
std::string encode(std::unordered_map<char, std::string> bitmap, 
                   std::string phrase);

// Does a simple search
void depth_first_search(node* root, huffman_cp &current,
                                      std::vector<huffman_cp> &bitstrings);
// Does a simple 2-pass encoding scheme
huffman_tree two_pass_huffman(std::string phrase);

// Simple decoding scheme
void decode(huffman_tree encoded_tree);

#endif
