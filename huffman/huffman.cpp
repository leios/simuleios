/*-------------huffman.cpp----------------------------------------------------//
*
* Purpose: Create a simple huffman tree, given a likelihood of receiving
*          certain letters
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include "huffman.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

/*
int main(){

    // encoding with 2-pass huffman
    std::string phrase ="Jack and Jill went up the hill to fetch a pail of water. Jack fell down and broke his crown and Jill came Tumbling after! \nWoo!";
    huffman_tree final_tree = two_pass_huffman(phrase);
    decode(final_tree);

    for (auto element : final_tree.internal){
        std::cout << element->weight << '\n';
    }

    for (auto element : final_tree.external){
        std::cout << element->key << '\n';
    }

}
*/

// creates nodes from vectors of characters and doubles
node_queue create_nodes(std::vector<char> &keys, std::vector<double> &weights){

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

// Overloaded create_nodes function
node_queue create_nodes(std::unordered_map<char, double> &keyweights){
    node_queue initial_nodes;

    node* tempnode;

    for (auto& key : keyweights){
        tempnode = new node(key.first, key.second);
        tempnode->left = nullptr;
        tempnode->right = nullptr;
        tempnode->parent = nullptr; 
        initial_nodes.push(tempnode);
        
    }
    return initial_nodes;

}

// Creates the simple binary tree
node* huffman(node_queue &initial_nodes, std::vector<node*> &internal,
              std::vector<node*> &external){

    node *node1, *node2, *node_parent;

    // Sort the vector by the weight
    while (initial_nodes.size() > 1){
        // Add two nodes together and point to the previous nodes
        node1 = initial_nodes.top();

        // Check to see if node is an internal or external node
        if (node1->key){
            // External node
            external.push_back(node1);
        }
        else{
            // internal node
            internal.push_back(node1);
        }

        initial_nodes.pop();
        node2 = initial_nodes.top();

        // Check to see if node is an internal or external node
        if (node2->key){
            // External node
            external.push_back(node2);
        }
        else{
            // internal node
            internal.push_back(node2);
        }

        initial_nodes.pop();

        node_parent = new node();

        node_parent->weight = node1->weight + node2->weight;
        node_parent->left = node1;
        node_parent->right = node2;

        node1->parent = node_parent;
        node2->parent = node_parent;

        initial_nodes.push(node_parent);

        //std::cout << initial_nodes.size() << '\n';
        
    }

    // Should be returning just the root node in initial_nodes
    // root node needs to be placed into internal nodes
    internal.push_back(initial_nodes.top());

    return initial_nodes.top();
    
}

// creates bit code by recreating the huffman tree
// sets length and code, itself
// Note: To be continued
std::unordered_map<char, std::string> create_bits(node* &root){

    std::vector<huffman_cp> bitstrings;
    std::unordered_map<char, std::string> bitmap;
    huffman_cp current;

    // Traverse the tree to create bits
    depth_first_search(root, current, bitstrings);

    for (size_t i = 0; i < bitstrings.size(); ++i){
        bitmap[bitstrings[i].key] = bitstrings[i].code;
        //std::cout << bitstrings[i].key << '\t' << bitstrings[i].code << '\n';
    }

    return bitmap;
}

// Does a simple search
void depth_first_search(node* &root, huffman_cp &current,
                                      std::vector<huffman_cp> &bitstrings){

    // Are we on a leaf node?
    if (!root->right && !root->left){
        current.key = root->key;
        bitstrings.push_back(current);
    }

    if (root->right){
        current.code += "0";
        depth_first_search(root->right, current, bitstrings);
        current.code.pop_back();
    }
    if (root->left){
        current.code += "1";
        depth_first_search(root->left, current, bitstrings);
        current.code.pop_back();
    }

}

// Does a simple search to regenerated node_queue from root
void depth_first_search(node* &root, node_queue &regenerated_nodes){

    // Filling the regenerated nodes 
    regenerated_nodes.push(root);

    if (root->right){
        depth_first_search(root->right, regenerated_nodes);
    }
    if (root->left){
        depth_first_search(root->left, regenerated_nodes);
    }

}


// does the encoding -- Assuming that all characters in phrase are already
// encoded in bitmap
std::string encode(std::unordered_map<char, std::string> &bitmap, 
                   std::string &phrase){

    std::cout << "phrase is: " << phrase << '\n';
    std::string encoded_phrase;

    for (size_t i = 0; i < phrase.size(); ++i){
        encoded_phrase += bitmap[phrase[i]];
    }

    std::cout << "encoded phrase is: " << encoded_phrase << '\n';
    return encoded_phrase;
}

// Does a simple 2-pass encoding scheme
huffman_tree two_pass_huffman(std::string &phrase){

    huffman_tree final_tree;
    final_tree.phrase = phrase;

    // Adding weights to the weightmap in our final huffman tree
    for (size_t i = 0; i < phrase.size(); ++i){
        if(final_tree.weightmap[phrase[i]]){
            final_tree.weightmap[phrase[i]] += 1;
        }
        else{
            final_tree.weightmap[phrase[i]] = 1;
        }
    }

    // finding size of alphabet
    final_tree.alphabet_size = final_tree.weightmap.size();

    // Creating initial external nodes
    node_queue initial_nodes = create_nodes(final_tree.weightmap);

    // Performing huffman algorithm

    // Creating internal and external node vectors
    // Number of internal nodes is # leaf nodes (alphabet size) - 1
    final_tree.internal.reserve(final_tree.alphabet_size - 1);
    final_tree.external.reserve(final_tree.alphabet_size);
    final_tree.root = huffman(initial_nodes, final_tree.internal, 
                              final_tree.external);

    // outputting the weights of internal and external nodes
    std::cout << "internal nodes are: " << '\n';
    for (auto node : final_tree.internal){
        std::cout << node->weight << '\n';
    }

    std::cout << '\n';
    std::cout << "external node are: " << '\n';

    for (auto node : final_tree.external){
        std::cout << node->key << '\t' << node->weight << '\n';
    }

    // creating our map from characters to bits
    final_tree.bitmap = create_bits(final_tree.root);

    for (auto& key : final_tree.bitmap){
        std::cout << key.first << '\t' << key.second << '\n';
    }

    // Doing the encoding of our phrase
    final_tree.encoded_phrase = encode(final_tree.bitmap, phrase);

    // finding size of alphabet
    //final_tree.alphabet_size = final_tree.bitmap.size();

    return final_tree;

}

// Simple decoding scheme
void decode(huffman_tree &encoded_tree){
    
    // Create decoding map
    std::unordered_map<std::string, char> decoding_map;
    for (auto& key : encoded_tree.bitmap){
        decoding_map[key.second] = key.first;
    }

    // Now we iterate through the string
    std::string temp_string, decoded_phrase;
    for (size_t i = 0; i < encoded_tree.encoded_phrase.size(); ++i){
        temp_string += encoded_tree.encoded_phrase[i];
        if (decoding_map[temp_string]){
            decoded_phrase += decoding_map[temp_string];
            temp_string = "";
        }
    }

    std::cout << "decoded phrase is: " << decoded_phrase << '\n';
}

