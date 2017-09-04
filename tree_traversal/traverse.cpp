/*-------------traverse.cpp---------------------------------------------------//
*
* Purpose: This file tests multiple ways to traverse a tree
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <stack>
#include <queue>

struct node{
    node *parent;
    node **children;
    int num_children;
    int ID;
};

// Function to create simple tree
void create_tree(node* &root, int num_row, int num_child);

// Function to do a depth-first search recursively
void DFS_recursive(node* &root);

// Function to do a depth-first search with a stack
void DFS_stack();

// Function to do a breadth-first search with a queue
void BFS_queue();

int main(){

    // Creating tree
    node *root;
    create_tree(root, 3, 3);
    std::cout << "Tree created!" << '\n';
    DFS_recursive(root);

}


// Function to create simple tree
void create_tree(node* &root, int num_row, int num_child){
    root = new node;
    root->ID = num_row;
    root->num_children = num_child;
    root->children = new node*[num_child];
    if (num_row > 0){
        for (int i = 0; i < num_child; ++i){
            create_tree(root->children[i], num_row - 1, num_child);
        }
    }
}

// Function to do a depth-first search recursively
void DFS_recursive(node* &root){
    if (!root){return;}
    std::cout << root->ID << '\n';
    for (int i = 0; i < root->num_children; ++i){
        DFS_recursive(root->children[i]);
    }
}
