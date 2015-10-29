#-------------demon.py---------------------------------------------------------#
#
#             demon.py
# 
# Purpose: Visualize the maxwell's demon code from earlier on stream!
#
#------------------------------------------------------------------------------#

import bpy
import numpy as np

#------------------------------------------------------------------------------#
# MAIN
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# SUBROUTINES
#------------------------------------------------------------------------------#

# goes through all the data! Woo!
def parse_data(num_part):
    array = [[]*8]*(num_part)
    i = 0
    offset = 0
    print("importing data from file")
    input = "file.dat"
    with open(input, 'r') as data:
        for line in data:
            if line != '\n':
                temp = [float(s) for s in line.split()]
                array[(i) % num_part] = temp
                i += 1
            if i % num_part == 0 and i != 0:
                place_spheres(array, num_part, i)
                #scene.update()

            
    return array

# Creates sphere material
def create_new_material (passedName,passedcolor):
    tempMat = bpy.data.materials.new(passedName)
    if tempMat != None:
        tempMat.diffuse_color = passedcolor
        tempMat.diffuse_shader = 'LAMBERT'
        tempMat.diffuse_intensity = 1.0
        tempMat.specular_color = (0.9,0.9,0.9)
        tempMat.specular_shader = 'COOKTORR'
        tempMat.specular_intensity = 0.5
        tempMat.use_transparency=False
        tempMat.alpha = 0.5
        tempMat.ambient = 0.3
        tempMat.emit = 0.2
    return tempMat

# places new sphere at given location
def new_sphere(diam, x, y, z, r, g, b, id):
    temp_sphere = bpy.ops.mesh.primitive_uv_sphere_add(segments = 64, 
                                                       ring_count = 32,
                                                       size = diam,
                                                       location = (x, y, z),
                                                       rotation = (0, 0, 0))
    ob = bpy.context.active_object
    ob.name = str(id)
    me = ob.data
    color = (r, g, b)
    mat = create_new_material("myNewMaterial",color)
    me.materials.append(mat)
    return temp_sphere

# places sphere duplicates around for fun!
def place_duplicates(x, y, z, id, ob = None):
    if not ob:
        ob = bpy.context.active_object
    obs = []
    sce = bpy.context.scene
        
    copy = ob.copy()
    copy.location = x,y,z
    copy.data = copy.data.copy()
    copy.name = str(id)
    obs.append(copy)
    
    for ob in obs:
        sce.objects.link(ob)
    
    #sce.update()

# function to place spheres in blender
def place_spheres(array, num_part, i):
    diam = 0.1

    #print(array)

    if i == num_part:
        for i in range(0, num_part):
            if i == 0:
                new_sphere(diam, array[i][0], array[i][1], array[i][2], 0, 0, 1,
                           array[i][7])
            else:
                place_duplicates(array[i][0], array[i][1], array[i][2], 
                                 array[i][7])
    else:
        move_spheres(array, num_part, (i / num_part) * 1)

# Function to moves spheres that are already there.
# Not currently working!
def move_spheres(array, num_part, frame):
    for i in range(num_part):
        bpy.context.scene.frame_set(frame)  
        print(str(array[i][7]))
        ob = bpy.context.scene.objects[str(array[i][7])]
        ob.location = (array[i][0], array[i][1], array[i][2])
        ob.keyframe_insert(data_path="location", index=-1)

scene = bpy.context.scene
parse_data(5)
scene.update()
#print (array)
