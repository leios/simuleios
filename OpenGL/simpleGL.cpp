/*-------------simpleGL.cpp---------------------------------------------------//
*
* Purpose: to test out some OpenGL stuff!
*
*   Notes: Triangle uses height for scaling along x, fix that maybe?
*          Enumerate all vertices in our triangle
*          scale across the surface of a sphere
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>

// GLEW must be defined first
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW next
#include <GLFW/glfw3.h>

#include "shader.h"
#include "vec.h"

// callback function for keys
void key_callback(GLFWwindow *window, int key, int scancode, int action, 
                  int mode);

// Function to define vertices and indices for a giant triangle
void create_triangle_vertex(std::vector<vec> &vertices, 
                            std::vector<triangle> &indices, int res);

// Function to recursively create triangle of triangles
void divide_triangle(std::vector<vec> &corners, std::vector<triangle> &indices, 
                     std::vector<vec> &vertices, int depth);

// Function to return a triangle number when provided an integer
int triangle_number(int n){
    return n * (n+1) / 2;
}

// defining shader strings to be built later
const GLchar* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n"
    "void main()\n"
    "{\n"
    "gl_Position = vec4(position.x, position.y, position.z, 1.0);\n"
    "}\0";
const GLchar* fragmentShaderSource = "#version 330 core\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    // Initializing glfw for window generation
    glfwInit();

    // Setting window variables
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Generating window instance
    GLFWwindow *window = glfwCreateWindow(800, 600, "Hello World", 
                                          nullptr, nullptr);
    if (window == nullptr){
        std::cout << "Window could not be created!" << '\n';
        glfwTerminate();
        return(-1);
    }

    glfwMakeContextCurrent(window);

    // GLEW manages function pointers for OpenGL
    // GL_TRUE gives us a more modern GLEW to work with
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK){
        std::cout << "Failed to initialize GLEW" << '\n';
    }

    // Setting viewport up so OpenGL knows what to draw to and where
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Read from LL to UR
    // Here, we map -1 < x < 1 in x and y to 800 x 600
    glViewport(0,0,width,height);

    // Registering closing callback function
    // glfwPollEvents later should be waiting for key callbacks
    glfwSetKeyCallback(window, key_callback);

    // Building / compiling shader program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR: vertex shader compilation failed!" << '\n';
    }

    // Now the same for the Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for fragment shader errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success){
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR: fragment shader compilation failed!" << '\n';
    }

    // Now to link the shaders together
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Checking for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success){
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR: program linking failed!" << '\n';
    }

    // Removing shaders after program is built
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Rendering a triangle.. first, we need to define the vertices
    // Note: GL assumes 3 dimensions, and likely parses the vertices by groups
    //       of three.
    //int max_depth = 10;
    std::vector<vec> vertices;
    //vertices.reserve(3 * pow(4,max_depth));
    std::vector<triangle> indices;
    //indices.reserve(pow(4,max_depth));
    create_triangle_vertex(vertices, indices, 1000);
    std::vector<vec> init_corners(3);
    init_corners = {
        vec(0,1,0),
        vec(-1,-1,0),
        vec(1,-1,0)
    };
    //divide_triangle(init_corners, indices, vertices, max_depth);

    // Now to request space on the GPU with Vertex Buffer Objects (VBO)
    // VAO is Vertex Array Object
    // Buffer has a unique ID
    GLuint VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Binding VAO first
    glBindVertexArray(VAO);

    // Specify the type of buffer:
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    // Now to do the copying
    // GL_STATIC_DRAW: No changes in vertices expected
    // GL_DYNAMIC_DRAW:  Regular changes expected
    // GL_STREAM_DRAW: Changes expected every step
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * 3 * sizeof(float), 
                 vertices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * 3 * sizeof(int), 
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 
                          3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Called a "game loop" continues until window needs to close
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // Setting window color and such:
        glClearColor(0.2f, 0.3f, 0.3f, 0.5f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Using the program we created
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        //glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawElements(GL_TRIANGLES, indices.size() * 3, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glfwSwapBuffers(window);
    }

    // termination
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// callback function for keys
void key_callback(GLFWwindow *window, int key, int scancode, int action, 
                  int mode){
    // esc = close window, set WindowShouldClose = True
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

// Function to define vertices and indices for a giant triangle
void create_triangle_vertex(std::vector<vec> &vertices, 
                            std::vector<triangle> &indices, int res){

    //std::cout << "making triangle stuff" << '\n';
    // assume triangle uses entire screen
    float height = 2.0 / (float)res;
    float startx = 0;
    float starty = 1.0;
    float x = startx;
    vec vertex;

    res++;

    // finding number of vertices
    int numvertex;
    int row = 1;
    int intcount = 0;

    // Defining all vertices
    numvertex = (res + 1) * (res) / 2.0;
    vertices.reserve(numvertex-1);
    //std::cout << "numvertex = " << numvertex << '\n';
    for (int i = 0; i < numvertex; ++i){
        vertex.x = x;
        vertex.y = starty;
        vertex.z = 0.0;

        vertices.push_back(vertex);
        x += height;

        if (i == intcount){
            startx -= 0.5 * height;
            x = startx;
            starty -= height;
            row++;
            intcount += row;
        }
    }

/*
    for (int i = 0; i < vertices.size(); i++){
        std::cout << i << '\n';
        print(vertices[i]);
    }
    std::cout << "done printing" << '\n';
*/

    // this is done by keeping track of every row and incrementing top and 
    // bottom counts
    int top_count, bot_count, j, row_count;
    row_count = 1;
    bot_count = 2;
    triangle temp_triangle;
    while (bot_count < numvertex){
        //std::cout << bot_count << '\n';
        top_count = triangle_number(row_count - 1)+1;
        j = 0;
        while(bot_count < triangle_number(row_count+1)){
            // upward-facing triangle
            if (j % 2 == 0){
                temp_triangle.ab = top_count-1;
                temp_triangle.bc = bot_count-1;
                temp_triangle.ca = bot_count;
                indices.push_back(temp_triangle);
                bot_count++;
            }
            // downward-facing triangle
            else{
                temp_triangle.ab = top_count-1;
                temp_triangle.bc = top_count;
                temp_triangle.ca = bot_count-1;
                indices.push_back(temp_triangle);
                top_count++;
            }
            j++;
        }
        bot_count++;
        row_count++;
    }

/*
    for (auto tri : indices){
        std::cout << tri.ab << '\t' << tri.bc << '\t' << tri.ca << '\n';
    }
*/
}

void divide_triangle(std::vector<vec> &corners, std::vector<triangle> &indices, 
                     std::vector<vec> &vertices, int depth){

    // Creating the triangle
    vertices.push_back(corners[0]);
    vertices.push_back(corners[1]);
    vertices.push_back(corners[2]);
    triangle temp_tri;
    temp_tri.ab = vertices.size() - 3;
    temp_tri.bc = vertices.size() - 2;
    temp_tri.ca = vertices.size() - 1;
    indices.push_back(temp_tri);

    std::vector<vec> temp_corners(3);

    //std::cout << vertices.size() << '\t' << indices.size() << '\n';
    //print(corners[0]);
    //print(corners[1]);
    //print(corners[2]);
    //std::cout << temp_tri.ab << '\t' << temp_tri.bc << '\t' 
    //          << temp_tri.ca << '\n';

    if (depth > 0){
        // Top triangle
        temp_corners[0] = corners[0];
        temp_corners[1] = (corners[0] + corners[1]) / 2.0;
        temp_corners[2] = (corners[0] + corners[2]) / 2.0;
        divide_triangle(temp_corners, indices, vertices, depth - 1);

        // bottom left
        temp_corners[0] = corners[1];
        temp_corners[1] = (corners[1] + corners[0]) / 2.0;
        temp_corners[2] = (corners[1] + corners[2]) / 2.0;
        divide_triangle(temp_corners, indices, vertices, depth - 1);

        // bottom center
        temp_corners[0] = (corners[1] + corners[2]) / 2.0;
        temp_corners[1] = (corners[0] + corners[1]) / 2.0;
        temp_corners[2] = (corners[0] + corners[2]) / 2.0;
        divide_triangle(temp_corners, indices, vertices, depth - 1);

        // bottom right
        temp_corners[0] = corners[2];
        temp_corners[1] = (corners[2] + corners[1]) / 2.0;
        temp_corners[2] = (corners[2] + corners[0]) / 2.0;
        divide_triangle(temp_corners, indices, vertices, depth - 1);

    }
    
}

