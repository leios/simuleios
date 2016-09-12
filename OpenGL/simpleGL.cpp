/*-------------simpleGL.cpp---------------------------------------------------//
*
* Purpose: to test out some OpenGL stuff!
*
*   Notes: Add a new vector for ints and indices
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
                            std::vector<GLuint> &indices, int res);

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
/*
    std::vector <GLfloat> vertices(15);
    vertices = {
    //GLfloat vertices[] ={
        0.5f, 0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        -0.5f, 0.5f, 0.0f,
        0.0f, 0.75f, 0.0f
    };
*/

    std::vector<vec>vertices(5);
    vertices = {
        vec(0.5, 0.5, 0.0),
        vec(0.5, -0.5, 0),
        vec(-0.5, -0.5, 0),
        vec(-0.5, 0.5, 0),
        vec(0, 0.75, 0)
    };

    std::vector <GLuint> indices(9);
    indices = {
    //GLuint indices[] = {
        0,1,3,
        1,2,3,
        0,3,4
    };

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
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * 3 *sizeof(float), 
                 vertices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int), 
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
        glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glfwSwapBuffers(window);
    }

    // termination
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();

}

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
                            std::vector<GLuint> &indices, int res){
    // assume triangle uses entire screen
    float height = 2 / res;
    float startx = 0;
    float starty = 1.0
    float x = startx;

    // finding number of vertices
    int numvertex = 0;
    int row = 1;
    int intcount = 0;

    // Defining all vertices
    numvertex = (res + 1) * (res) / 2.0;
    for (int i = 0; i < numvertex; ++i){
        vertices[i].x = x;
        vertices[i].y = starty;
        vertices[i].z = 0.0;
        indices[i] = i;

        x += 0.5 * height;

        if (i == intcount){
            startx -= 0.5 * height;
            x = startx;
            starty -= height;
            row++;
            intcount += row;
        }
    }

/*
    // defining all indices
    row = 1;
    int index = 0;
    for (int i = 0 i < indices.size(); ++i){
        // Upright triangles on row
        // Upside down triangles on row

        indices[i] = index;
    }
*/
}
