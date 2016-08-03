/*-------------vitter.cpp-----------------------------------------------------//
*
* Purpose: Create an adaptive huffman tree, given a likelihood of receiving
*          certain letters
*
*   Notes: Adding elements if already in tree has not been completed
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<vector>
#include<algorithm>
#include<unordered_map>
#include "huffman.h"

struct block{
    int internal_start;
    int external_start;
  
    block() : internal_start(-1), external_start(-1) {}
    block(int is, int es) : internal_start(is), external_start(es) {}
    std::vector<node *> internal, external;
};

// Function to slide and increment object along tree
void slideandincrement(block &blockset);

// Function to generate dynamic tree
void vitter(std::string &phrase, huffman_tree &tree);

// Function to test to see whether a new character is in the tree
bool in_tree(std::vector<node*> &external, char new_key);

// Function to extend NYT or 0 node
node* extend_node(huffman_tree &tree, char new_key);

// Function to swap two external nodes of same weight
node* swap(huffman_tree &tree, char new_key);

// Function to find node from its bitstring
node* find_node(huffman_tree tree, std::string guide_to_node);

// Function to add external node to external vector of nodes
void add_external(huffman_tree tree, node *external_to_add, 
                  std::vector<node*> &external);

// Function to add internal node to internal vector of nodes
void add_internal(huffman_tree tree, node *internal_to_add, 
                  std::vector<node*> &internal);

// Function to find the leader of a block
// Function written by Gustorn
// Note: requires specific ordering: [W1L, W1, W1, W2L, W2, W2, W3L, W3, W3]
node* find_leader(const std::vector<node*>& internal, node* block_member) {
    return *std::find_if(std::begin(internal), 
                         std::end(internal), 
                         [block_member](node *n) 
                             { return n->weight == block_member->weight; });
}

// Function to find insertion index
int find_insert_index(std::vector<node*> &internal, node* block_member);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){
    std::string phrase = "Jack and Jill went up the hill to fetch a pail of water. Jack fell down and broke his crown and Jill came Tumbling after!";

    huffman_tree tree = two_pass_huffman(phrase);

    // Create a map so we may call block with integers
    std::unordered_map<int, block> block_positions;

    // find the block leaders
    // Internal first
    std::cout << "internal node leader weight is: " << '\n';
    for (auto in_node : tree.internal){
        node* leader;
        leader = find_leader(tree.internal, in_node);
        std::cout << leader->weight << '\n';
    }

    // External next
    std::cout << "external node leader weight is: " << '\n';
    for (auto ex_node : tree.external){
        node* leader;
        leader = find_leader(tree.external, ex_node);
        std::cout << leader->weight << '\n';
    }

    // Checking our in_tree command
    bool in_tree_check = in_tree(tree.external, 'q');

    std::cout << "1 if in: " << in_tree_check << '\n';

    // Checking the vitter function
    huffman_tree vitter_tree;
    vitter_tree.NYT = new node();
    std::string vitter_phrase = "qqJ";
    vitter(vitter_phrase, vitter_tree);
    
    
}

// Function to slide and increment object along tree
void slideandincrement(block &blockset){
}

// Function to generate dynamic tree
void vitter(std::string &phrase, huffman_tree &tree){

    for (size_t i = 0; i < phrase.size(); ++i){
        // Element already in tree
        if (in_tree(tree.external, phrase[i])){
            std::cout << "Element already in tree." << '\n';
            swap(tree, phrase[i]);
        }

        // Element must be added to tree
        else{
            std::cout << "Element not found in tree... adding..." << '\n';
            node *q = extend_node(tree, phrase[i]);
        }
        
    }
}

// Function to test to see whether a new character is in the tree
bool in_tree(std::vector<node*> &external, char new_key){
    return std::find_if(std::begin(external), std::end(external),
                        [new_key](node *n) 
                            {return n->key == new_key;}) != std::end(external);

}

// Function to extend the NYT or 0 node
// Will replace the NYT node with a new NYT node (left) and external (right)
node* extend_node(huffman_tree &tree, char new_key){

    node *node_parent, *node_external;

    // Creating new parent node
    node_parent = new node();
    node_parent->weight = 0;
    node_parent->left = tree.NYT;

    // Creating new external node with given character
    node_external = new node();
    node_external->weight = 0;
    node_external->key = new_key;
    node_external->parent = node_parent;

    // placing the NYT on the left, and new external node on right
    node_parent->right = node_external;

    if (tree.NYT->parent){
        tree.NYT->parent->left = node_parent;
    }

    tree.NYT->parent = node_parent;

    // Add the new nodes to external and internal
    add_external(tree, node_external, tree.external);

    add_internal(tree, node_parent, tree.internal);

    return node_parent;
}

// Function to swap two external nodes of same weight
node* swap(huffman_tree &tree, char new_key){
    // Find the external node to swap
    // This is inefficient, but will probably work
    // We will be using the bitmap to transverse the tree to get our node
    std::string guide_to_node = tree.bitmap[new_key];

    node *found_node = find_node(tree, guide_to_node);

    // Do the swapping

    return found_node->parent;
}

node* find_node(huffman_tree tree, std::string guide_to_node){
    node *found_node = tree.root;

    // Traverse tree to find node
    for (int i = 0; i < guide_to_node.size(); ++i){
        if (guide_to_node[i] == 1){
            found_node = found_node->left;
        }

        // Only other case is "0"
        else{
            found_node = found_node->right;
        }
    }

    return found_node;

}

// NOTE: A more clever inserting scheme should be implemented.
//       Here, we are simply adding the nodes to the vector in front of the 
//       leader of the following block (weight+1)

// Function to add external node to external vector of nodes
void add_external(huffman_tree tree, node *external_to_add, 
                  std::vector<node*> &external){
    int index;
    index = find_insert_index(external, external_to_add);

    // Inserting into external vector
    external.insert(external.begin() + index, external_to_add);
    
}

// Function to add internal node to internal vector of nodes
void add_internal(huffman_tree tree, node *internal_to_add, 
                  std::vector<node*> &internal){
    int index;
    index = find_insert_index(internal, internal_to_add);

    // Inserting into external vector
    internal.insert(internal.begin() + index, internal_to_add);

}

// Function to find insertion index
int find_insert_index(std::vector<node*> &internal, node* block_member){
    int index = 0;
    for (int i = 0; i < internal.size(); ++i){
        if (internal[i]->weight - 1 == block_member->weight){
            index = i;
            break;
        }
    }

    return index;
}

