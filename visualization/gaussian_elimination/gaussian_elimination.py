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

import sys
import os
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
from blender_render import *

def gaussian_elimination(A):
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
        row += 1

def gauss_jordan_elimination(A):

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

        row += 1

# Reads in 4 values, outputs 3 angles
def find_angle_plane(eqn):
    angles = [0.0,0.0,0.0]
    angles[0] = eqn[3] / eqn[0]
    angles[1] = eqn[3] / eqn[1]
    angles[2] = eqn[3] / eqn[2]

    return angles

def rotate_plane(plane_string, angles, timestep):
    bpy.data.objects[plane_string].rotation_euler[0] = angles[0]
    bpy.data.objects[plane_string].rotation_euler[1] = angles[1]
    bpy.data.objects[plane_string].rotation_euler[2] = angles[2]

# Reads in array and finds appropriate angle for each plane
def visualize_matrix(A, scene, timestep):
    for i in range(3):
        temp_vector = A[i, :]
        angles = find_angle_plane(temp_vector)
        plane_string = ""
        if (i == 0):
            plane_string = "plane1"
        elif (i == 1):
            plane_string = "plane2"
        elif (i == 2):
            plane_string = "plane3"

        rotate_plane(plane_string, angles, timestep)
        bpy.context.scene.objects[plane_string].keyframe_insert(
            data_path='rotation_eluer', frame=(timestep*60))

# defining matrix
A = np.array([[2., 3, 4, 6],
              [1, 2, 3, 4],
              [3, -4, 0, 10]])
print("Original Matrix")
print(A)

gaussian_elimination(A)
print("Gaussian Elimination Matrix")
print(A)

gauss_jordan_elimination(A)
print("Gauss Jordan Elimination Matrix")
print(A)

x_soln = A[0,3]
y_soln = A[1,3]
z_soln = A[2,3]

print(x_soln, y_soln, z_soln)


num = 10
scene = bpy.context.scene
scene = def_scene(75,scene)
remove_obj(scene)

define_axes(20)

new_plane(x_soln,y_soln,z_soln, 0,0,0,         0,0,1,0.5, 10, "plane1")
new_plane(x_soln,y_soln,z_soln, 0,0.5*np.pi,0, 1,0,0,0.5, 10, "plane2")
new_plane(x_soln,y_soln,z_soln, 0.5*np.pi,0,0, 0,1,0,0.5, 10, "plane3")

visualize_matrix(A, scene, 1)

new_sphere(0.5,x_soln,y_soln,z_soln, 0, 0, 1, 1)

# Adding in extra function for determinant visualization
bpy.data.scenes["Scene"].frame_end = num
scene.update()
render_movie(scene)
