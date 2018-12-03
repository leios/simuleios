#-------------gaussian_elimination.py------------------------------------------#
#
#             gaussian_elimination.py
# 
# Purpose: Visualize gaussian elimination!
#
#   Notes: This code can be run by using the following command:
#              blender -b -P gaussian_elimination.py
#          To show on stream, use:
#              mplayer out.mp4 -vo x11
#
#------------------------------------------------------------------------------#
import bpy
import numpy as np
import random
import math

import sys
import os
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
from blender_render import *

def gaussian_elimination(A, visualize, step, soln):
    rows = A.shape[0]
    cols = A.shape[1]

    # Row index
    row = 0

    for col in range(cols-2):
        max_index = np.argmax(abs(A[:,col]))

        if (A[max_index, col] == 0):
            print("Matrix is singlular!")

        A[[max_index,row]] = A[[row,max_index]]

        for i in range(row+1, rows):
            fraction = A[i, col]/A[row,col]
            for j in range(col+1, cols):
                A[i,j] -= A[row,j]*fraction
            A[i,col] = 0
            if (visualize):
                visualize_matrix(A, scene, step, soln, False)
                visualize_matrix(A, scene, step, soln, True)
        row += 1

def gauss_jordan_elimination(A, visualize, step, soln):

    rows = A.shape[0]
    cols = A.shape[1]

    row = 0
    for col in range(cols-1):
        if (A[row, col] != 0):
            for i in range(cols-1, col-1, -1):
                A[row,i] /= A[row, col]

        for i in range(row):
            for j in range(cols-1, col-1, -1):
                A[i,j] -= A[i,col]*A[row,j]

            if (visualize):
                visualize_matrix(A, scene, step, soln, False)
                visualize_matrix(A, scene, step, soln, True)

        row += 1

def mag(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1])

def mag3d(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def dot(a,b):
    return (a[0]*b[0] + a[1]*b[1] +a[2]*b[2])

# Reads in 4 values, outputs 3 angles
def find_angle_plane(eqn, loc):
    angles = [0.0,0.0,0.0]
    for i in range(3):
        plane = [1, 0, 0]
        if (i == 1):
            plane = [0, 1, 0]
        if (i == 2):
            plane = [0, 0, 1]
        angles[i] = math.acos(dot(eqn, plane)/(mag3d(eqn)*mag3d(plane)))

    angles[1] -= math.pi/2
    angles[2] -= math.pi/2
    return angles

def rotate_plane(plane_string, angles):
    bpy.data.objects[plane_string].rotation_euler[0] = angles[0]
    bpy.data.objects[plane_string].rotation_euler[1] = angles[1]
    bpy.data.objects[plane_string].rotation_euler[2] = angles[2]

# Reads in array and finds appropriate angle for each plane
def visualize_matrix(A, scene, timestep, soln, is_pause):
    for i in range(3):
        temp_vector = A[i, :]
        angles = find_angle_plane(temp_vector, soln)
        plane_string = ""
        if (i == 0):
            plane_string = "plane1"
        elif (i == 1):
            plane_string = "plane2"
        elif (i == 2):
            plane_string = "plane3"

        rotate_plane(plane_string, angles)
        if (is_pause):
            bpy.context.scene.objects[plane_string].keyframe_insert(
                data_path='rotation_euler', frame=(timestep[0]*30))
        else:
            bpy.context.scene.objects[plane_string].keyframe_insert(
                data_path='rotation_euler', frame=(timestep[0]*30))
        timestep[0] += 1

# Function to fade in all planes and solution point
def fade_objects(scene, timestep, fade_in):
    objects = ["plane1", "plane2", "plane3", "1"]
    for obj in objects:
        bpy.data.materials[obj].keyframe_insert(
            data_path='alpha', frame=(timestep[0]*30))
        timestep[0] += 1
        if (fade_in):
            if (obj != "1"):
                bpy.data.materials[obj].alpha = 0.75
            else:
                bpy.data.materials[obj].alpha = 1
        else:
            bpy.data.materials[obj].alpha = 0
        bpy.data.materials[obj].keyframe_insert(
            data_path='alpha', frame=(timestep[0]*30))

# defining matrix
A = np.array([[2., 3, 4, 6],
              [1, 2, 3, 4],
              [3, -4, 0, 10]])
print("Original Matrix")
print(A)

gaussian_elimination(A, False, [0], [0,0,0])
print("Gaussian Elimination Matrix")
print(A)

gauss_jordan_elimination(A, False, [0], [0,0,0])
print("Gauss Jordan Elimination Matrix")
print(A)

x_soln = A[0,3]
y_soln = A[1,3]
z_soln = A[2,3]
soln_vec = [x_soln, y_soln, z_soln]

print(x_soln, y_soln, z_soln)

num = 60
scene = bpy.context.scene
scene = def_scene(75,scene)
remove_obj(scene)

define_axes(30)

new_plane(x_soln,y_soln,z_soln, 0,0,0,         0,0,1,0, 10, "plane1")
new_plane(x_soln,y_soln,z_soln, 0,0.5*np.pi,0, 1,0,0,0, 10, "plane2")
new_plane(x_soln,y_soln,z_soln, 0.5*np.pi,0,0, 0,1,0,0, 10, "plane3")

new_sphere(0.3,x_soln,y_soln,z_soln, 0, 0, 1, 1)
bpy.data.materials["1"].use_transparency = True
bpy.data.materials["1"].alpha = 0

step = [0]

fade_objects(scene, step, True)

A= np.array([[2., 3, 4, 6],
              [1, 2, 3, 4],
              [3, -4, 0, 10]])
visualize_matrix(A, scene, step, soln_vec, False)
visualize_matrix(A, scene, step, soln_vec, False)
gaussian_elimination(A, True, step, soln_vec)
gauss_jordan_elimination(A, True, step, soln_vec)
fade_objects(scene, step, False)

# Adding in extra function for determinant visualization
bpy.data.scenes["Scene"].frame_end = step[0]*30
#bpy.data.scenes["Scene"].frame_end = 10
scene.update()
set_render_options(scene, 1920, 1080)
#render_movie(scene)
