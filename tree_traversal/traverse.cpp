/*-------------traverse.cpp---------------------------------------------------//
*
* Purpose: This file tests multiple ways to traverse a tree
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <stack>
#include <queue>
#include <vector>

struct node{
    std::vector<node> children;
    int ID;
};

// Function to create simple tree
void create_tree(node& root, int num_row, int num_child);

// Function to do a depth-first search recursively
void DFS_recursive(const node& root);

// Function to do a depth-first search with a stack
void DFS_stack();

// Function to do a breadth-first search with a queue
void BFS_queue();

int main(){

    // Creating tree
    node root;
    create_tree(root, 3, 2);
    std::cout << "Tree created!" << '\n';
    DFS_recursive(root);

}

// Function to create simple tree
void create_tree(node& root, int num_row, int num_child){
    root.ID = num_row;
    if (num_row == 0){
        return;
    }
    root.children.reserve(num_child);
    for (int i = 0; i < num_child; ++i){
        node child;
        create_tree(child, num_row - 1, num_child);
        root.children.push_back(child);
    }
}

// Function to do a depth-first search recursively
void DFS_recursive(const node& root){
    if (root.children.size() == 0){
        return;
    }
    std::cout << root.ID << '\n';
    for (int i = 0; i < root.children.size(); ++i){
        DFS_recursive(root.children[i]);
    }
}
