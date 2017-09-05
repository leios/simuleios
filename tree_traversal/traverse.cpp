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
void DFS_stack(const node& root);

// Function to do a breadth-first search with a queue
void BFS_queue(const node& root);

int main(){

    // Creating tree
    node root;
    create_tree(root, 3, 3);
    std::cout << "Tree created!" << '\n';
    //DFS_recursive(root);
    //DFS_stack(root);
    BFS_queue(root);

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

// Function to do a depth-first search with a stack
void DFS_stack(const node& root){

    std::stack<node> s;
    s.push(root);
    node temp;

    while (s.size() > 0){
        std::cout << s.top().ID << '\n';
        temp = s.top();
        s.pop();
        for (int i = 0; i < temp.children.size(); ++i){
            s.push(temp.children[i]);
        }
    }
}

// Function to do a breadth-first search with a queue
void BFS_queue(const node& root){
    std::queue<node> q;
    q.push(root);
    node temp;

    while (q.size() > 0){
        std::cout << q.front().ID << '\n';
        temp = q.front();
        q.pop();
        for (int i = 0; i < temp.children.size(); ++i){
            q.push(temp.children[i]);
        }
    }
}
