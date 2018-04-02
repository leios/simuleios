# This is for the PriorityQueue
using DataStructures

# Note: We can probably use ints instead of doubles for weights
struct Node
    right::Node
    left::Node
    parent::Node
    weight::Int64
    key::Char
end

# This will search through the tree to create bitstrings for each character
function dfs(n::Node)
end

# This outputs huffman tree to generate dictionary for encoding
function create_tree(phrase::String)
    # creating weights
    weights = PriorityQueue()
    for i in phrase
        if (haskey(weights, i))
            weights[i] += 1
        else
            weights[i] = 1
        end
    end

    # Creating all nodes to iterate through
    nodes = PriorityQueue(Node, Int64)
    while(length(weights) > 1)
        weight = peek(weights)[2]
        key = dequeue!(weights)
        temp_node = Node(Node(), Node(), Node(), weight, key)
        enqueue!(nodes, temp_node, weight)
    end

    while(length(nodes) > 2)
        node1 = dequeue!(nodes)
        node2 = dequeue!(nodes)
        temp_node = Node(node1, node2, Node(), node1.weight + node2.weight, "")
        node1.parent = temp_node
        node2.parent = temp_node
        enqueue!(nodes, temp_node, temp_node.weight)
    end

    huffman_tree = dequeue!(nodes)

end

# This outputs encoding Dict to be used for encoding
function create_dict(n::Node)
end

function encode(map::Dict{String, String}, phrase::String)
end

function decode(map::Dict{String, String}, bitstring::String)
end

function two_pass_huffman(phrase::String)
    create_tree(phrase)
end

two_pass_huffman("hello world")
