# This is for the PriorityQueue
using DataStructures

# creating abstract type for binary trees
abstract type BT end
type Empty <: BT end

type Node <: BT
    right::BT
    left::BT
    weight::Int64
    key::String
    bitstring::String
end

# This will search through the tree to create bitstrings for each character
# This outputs encoding Dict to be used for encoding
function create_codebook(n::Node)
    s = Stack(Node)
    push!(s, n)

    codebook = Dict{Char, String}()
 
    while (length(s) > 0)
        temp = pop!(s)
        if (typeof(temp.left) != Empty)
            temp.left.bitstring = string(temp.bitstring, "0")
            push!(s, temp.left)
        end
        if (typeof(temp.right) != Empty)
            temp.right.bitstring = string(temp.bitstring, "1")
            push!(s, temp.right)
        end
        if (typeof(temp.right) == Empty &&
            typeof(temp.left) == Empty)
            codebook[temp.key[1]] = temp.bitstring
        end
    end

    return codebook
end

# This outputs huffman tree to generate dictionary for encoding
function create_tree(phrase::String)

    # creating weights
    weights = PriorityQueue()
    for i in phrase
        temp_string = string(i)
        if (haskey(weights, temp_string))
            weights[temp_string] += 1
        else
            weights[temp_string] = 1
        end
    end

    # Creating all nodes to iterate through
    nodes = PriorityQueue{Node, Int64}()
    while(length(weights) > 0)
        weight = peek(weights)[2]
        key = dequeue!(weights)
        temp_node = Node(Empty(), Empty(), weight, key, "")
        enqueue!(nodes, temp_node, weight)
    end

    while(length(nodes) > 1)
        node1 = dequeue!(nodes)
        node2 = dequeue!(nodes)
        temp_node = Node(node1, node2, node1.weight + node2.weight, "", "")
        enqueue!(nodes, temp_node, temp_node.weight)
    end

    huffman_tree = dequeue!(nodes)
    return huffman_tree

end

function encode(codebook::Dict{Char, String}, phrase::String)
    final_bitstring = ""
    for i in phrase
        final_bitstring = string(final_bitstring, codebook[i])
    end

    return final_bitstring
end

function decode(codebook::Dict{Char, String}, bitstring::String)
end

function two_pass_huffman(phrase::String)
    huffman_tree = create_tree(phrase)
    codebook = create_codebook(huffman_tree)
    println(codebook)
    bitstring = encode(codebook, phrase)
    println(bitstring)
end

two_pass_huffman("hello")
