/*-------------huffman.cpp----------------------------------------------------//
*
* Purpose: Create a simple huffman tree, given a likelihood of receiving
*          certain letters
*
*   Notes: Binary tree not implement yet
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<vector>
#include<bitset>
#include<queue>

// Create the binary tree
struct node{
    char key;
    double weight;

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
    std::bitset<128> code;
    int length;
    char key;
};

using node_queue = std::priority_queue<node*,std::vector<node*>,node_comparer>;

// creates nodes from vectors of characters and doubles
node_queue create_nodes(std::vector<char> keys, std::vector<double> weights);

// Creates the simple binary tree
node* huffman(node_queue &initial_nodes);

// creates bit code
std::vector<huffman_cp> create_bits(node* root);

// does the encoding
void encode(std::vector<huffman_cp> bitstring);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Create vector of weights and keys
    std::vector<char> keys = { 'A', 'B' , 'C', 'D'};
    std::vector<double> weights = { 1, 2, 1, 3};

    node_queue initial_nodes = create_nodes(keys, weights);
    node *root = huffman(initial_nodes);

    std::cout << root->weight << '\n';
    std::cout << root->left->weight << '\n';

    std::vector<huffman_cp> bitstrings = create_bits(root);

/*
    int size = initial_nodes.size();
    for (size_t i = 0; i < size; ++i){
        std::cout << initial_nodes.top()->key << '\t'
                  << initial_nodes.top()->weight << '\n';
        initial_nodes.pop();
    }
*/

}

// creates nodes from vectors of characters and doubles
node_queue create_nodes(std::vector<char> keys, std::vector<double> weights){

    node_queue initial_nodes;
    //initial_nodes.reserve(keys.size());

    node* tempnode;

    for (size_t i = 0; i < keys.size(); ++i){
        tempnode = new node(keys[i], weights[i]);
        tempnode->left = nullptr;
        tempnode->right = nullptr;
        tempnode->parent = nullptr; 
        initial_nodes.push(tempnode);
        
    }
    return initial_nodes;
}

// Creates the simple binary tree
node* huffman(node_queue &initial_nodes){

    node *node1, *node2, *node_parent;

    // Sort the vector by the weight
    while (initial_nodes.size() > 1){
        // Add two nodes together and point to the previous nodes
        node1 = initial_nodes.top();
        initial_nodes.pop();
        node2 = initial_nodes.top();
        initial_nodes.pop();

        node_parent = new node();

        node_parent->weight = node1->weight + node2->weight;
        node_parent->left = node1;
        node_parent->right = node2;

        initial_nodes.push(node_parent);

        std::cout << initial_nodes.size() << '\n';
        
    }

    return initial_nodes.top();
    
}

// creates bit code by recreating the huffman tree
// sets length and code, itself
// Note: To be continued
std::vector<huffman_cp> create_bits(node* root){

    std::vector<huffman_cp> bitstrings;
    // Traverse the tree to create bits

    return bitstrings;
}

// does the encoding
void encode(std::vector<huffman_cp> bitstrings){
}


