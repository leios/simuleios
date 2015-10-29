#-------------demon.py---------------------------------------------------------#
#
#                        demon.py
#
# Purpose: Visualize the maxwell's demon code from earlier on stream!
#
#
# Hint: at the moment the script is still slow
#       takes about 5 minutes to import 5000 frames with 300 particles each
#       the expensive call are the keyframes
#       there must be / is  a better method to import the ipo curves directly
#------------------------------------------------------------------------------#

import bpy
import numpy as np
import itertools as it

#------------------------------------------------------------------------------#
# MAIN
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# SUBROUTINES
#------------------------------------------------------------------------------#

def chunk(iterable, n):
    iterator = iter(iterable)
    while True:
        chunk = it.islice(iterator, n)
        try:
            first = next(chunk)
        except StopIteration:
            return
        yield it.chain((first,), chunk)

# goes through all the data! Woo!
def parse_data(num_part):
        print("importing data from file")
        input = "/tmp/out.dat"
        with open(input, 'r') as data:
            chunks = chunk(data, num_part)
            place_spheres(next(chunks))

            for i, particle_set in enumerate(chunks):
                particles = ([float(s) for s in line.split()] for line in particle_set)
                if i % 100 == 0:
                    print("Adding keyframe {}".format(i))
                move_spheres(particles, i)

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
        copy.name = str(id)
        obs.append(copy)
        for ob in obs:
                sce.objects.link(ob)


# function to place spheres in blender
def place_spheres(particles):
        diam = 0.1
        for i, line in enumerate(particles):
            particle = [float(s) for s in line.split()]
            if i == 0:
                new_sphere(diam, particle[0], particle[1], particle[2], 0, 0, 1, particle[7])
            else:
                place_duplicates(particle[0], particle[1], particle[2], particle[7])



# Function to moves spheres that are already there.
def move_spheres(particles, frame):
        bpy.context.scene.frame_set(frame)
        current_frame = bpy.context.scene.frame_current
        for particle in particles:
                bpy.context.scene.objects[str(particle[7])].location = (particle[0], particle[1], particle[2])
                bpy.context.scene.objects[str(particle[7])].keyframe_insert(data_path='location', frame=(current_frame))

scene = bpy.context.scene
parse_data(300)
scene.update()
