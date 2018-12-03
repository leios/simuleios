#-------------blender_render.py------------------------------------------------#
#
#             blender_render.py
# 
# Purpose: Visualizing 3D objects with blender
#
#   Notes: This code can be run by using the following command:
#              blender -b -P blender_render.py
#          To show on stream, use:
#              mplayer out.mp4 -vo x11
#
#------------------------------------------------------------------------------#

import bpy
import numpy as np
import random

# Function to create a new plane
def new_plane(x, y, z, rx, ry, rz, r, g, b, a, size, name):
    plane = bpy.ops.mesh.primitive_plane_add(location=(x, y, z),
                                             rotation=(rx, ry, rz),
                                             radius = size)

    ob = bpy.context.active_object
    ob.name = name
    me = ob.data
    color = (r, g, b)
    mat = create_new_material(ob.name, color)
    mat.use_transparency = True
    mat.alpha = a
    me.materials.append(mat)


# This plots and x, y , and z axis
def define_axis(axis_type, length):
    if (axis_type == "x"):
        temp_axis = bpy.ops.mesh.primitive_cylinder_add(radius = 0.01,
                        rotation=(np.pi*0.5, 0, 0))
    elif (axis_type == "y"):
        temp_axis = bpy.ops.mesh.primitive_cylinder_add(radius = 0.01,
                        rotation=(0, np.pi*0.5, 0))
    elif (axis_type == "z"):
        temp_axis = bpy.ops.mesh.primitive_cylinder_add(radius = 0.01,
                        rotation=(0, 0, 0))
    else:
        print("Not real axis!")


    ob = bpy.context.active_object
    ob.scale[2] = length
    ob.name = axis_type + "axis"
    me = ob.data
    color = (1, 1, 1)
    mat = create_new_material(ob.name, color)
    me.materials.append(mat)

# Defines all 3 axes at once
def define_axes(length):
    define_axis("x", length)
    define_axis("y", length)
    define_axis("z", length)


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
        tempMat.keyframe_insert(data_path="diffuse_color", frame=1, index=-1)
    return tempMat

# places new sphere at given location
def new_sphere(diam, x, y, z, r, g, b, id):
    temp_sphere = bpy.ops.mesh.primitive_uv_sphere_add(segments = 32, 
                                                       ring_count = 16,
                                                       size = diam,
                                                       location = (x, y, z),
                                                       rotation = (0, 0, 0))
    ob = bpy.context.active_object
    ob.name = str(id)
    me = ob.data
    color = (r, g, b)
    mat = create_new_material(ob.name, color)
    me.materials.append(mat)
    return temp_sphere

# places sphere duplicates around for fun!
def place_duplicates(x, y, z, id, obid):
    ob = bpy.data.objects[obid]
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
# colors based on the sphere's velocity
def place_spheres(array, num_part, i):
    diam = 0.05

    # determine the final velocities for coloring
    vel_max = 0
    vel_min = 1000
    for i in range (num_part):
        vel = np.sqrt((array[i][3] * array[i][3]) + (array[i][4] * array[i][4])
                      + (array[i][5] * array[i][5]))
        if vel > vel_max:
            vel_max = vel
        if vel < vel_min:
            vel_min = vel

    vel_diff = vel_max - vel_min

    for i in range(0, num_part):
        vel = np.sqrt((array[i][3] * array[i][3]) + (array[i][4] * array[i][4])
                      + (array[i][5] * array[i][5]))

        ratio = (vel - vel_min) / vel_diff

        new_sphere(diam, array[i][0], array[i][1], array[i][2], 
                   0, 0, 1, array[i][7])
    return (vel_max, vel_min)

# Function to moves spheres that are already there.
def move_spheres(array, num_part, frame, max_vel, min_vel):
    bpy.context.scene.frame_set(frame)
    offset = int(frame * num_part - num_part)
    current_frame = bpy.context.scene.frame_current
    for i in range(offset,num_part+offset):
        vel = np.sqrt((array[i][3] * array[i][3]) 
              + (array[i][4] * array[i][4])
              + (array[i][5] * array[i][5]))
        diff_vel = max_vel - min_vel
        ratio = (vel - min_vel) / diff_vel
        mat = bpy.data.materials[str(array[i][7])]
        mat.diffuse_color = ( 0,0,1)
        mat.keyframe_insert(data_path="diffuse_color", frame=frame, index=-1)
        bpy.context.scene.objects[str(array[i][7])].location =  \
            (array[i][0],array[i][1],array[i][2])
        bpy.context.scene.objects[str(array[i][7])].keyframe_insert(
            data_path='location', frame=(current_frame))

# This function assumes that the balls have already been moved for this timestep
def move_lines(connectome, frame):
    for i in connectome:
        curve = bpy.data.curves["bc"+str(i[0])].splines[0]
        curve.bezier_points[0].co = \
            bpy.context.scene.objects[str(i[1])].location
        curve.bezier_points[0].keyframe_insert(data_path="co", 
                                               frame=frame, index=-1)
        curve.bezier_points[1].co = \
            bpy.context.scene.objects[str(i[2])].location

        curve.bezier_points[1].keyframe_insert(data_path="co", 
                                               frame=frame, index=-1)

# Creates the cage material
def create_cage (passedName):
    cageMat = bpy.data.materials.new(passedName)
    cageMat.type = 'WIRE'
    cageMat.diffuse_color = (1,1,1)
    cageMat.diffuse_shader = 'FRESNEL'
    cageMat.diffuse_intensity = 1
    cageMat.specular_color = (1,1,1)
    cageMat.use_diffuse_ramp = True
    ramp = cageMat.diffuse_ramp
    #(pt_location_on_ramp, (r,g,b,dens_at_pt))
    values = [(0.0, (1,1,1,1)), (1.0, (1,1,1,1))]
    for n,value in enumerate(values):
        ramp.elements.new((n+1)*0.2)
        elt = ramp.elements[n]
        (pos, color) = value
        elt.position = pos
        elt.color = color
    cageMat.diffuse_ramp_input = 'RESULT'
    return cageMat

# Creates cage at location
def cage_set(Box_length, x, y, z, id, make_frame):
    ccube = bpy.ops.mesh.primitive_cube_add(location=(x,y,z),
                                            radius = Box_length / 2)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    me = ob.data
    ob.name = id

    bpy.data.objects[id].hide = True
    bpy.data.objects[id].hide_render = True
    bpy.data.objects[id].keyframe_insert("hide_render", 
                                                 frame=make_frame-1)
    bpy.data.objects[id].keyframe_insert("hide", 
                                                 frame=make_frame-1)
    bpy.data.objects[id].hide = False
    bpy.data.objects[id].hide_render = False
    bpy.data.objects[id].keyframe_insert("hide_render", 
                                                 frame=make_frame)
    bpy.data.objects[id].keyframe_insert("hide", 
                                                 frame=make_frame)

    mat = create_cage('MaterialCage')
    me.materials.append(mat)
    return ccube

# Removes objects in scene
def remove_obj( scene ):
    for ob in scene.objects: 
        if ob.name !='Camera' and ob.name != 'Lamp':
            scene.objects.unlink( ob )

#defining our scene
def def_scene(box_length, scene):

    # Camera stuff
   
    '''
    # skewed angle
    x_cam = 2.2
    y_cam = 2.75
    z_cam = 1.43
    r_camx = 70
    r_camy = 0
    r_camz = 145

    # Side angle
    x_cam = 0
    y_cam = 0.5
    z_cam = 4
    r_camx = 0
    r_camy = 0
    r_camz = 0

    # skewed angle * 0.5
    x_cam = 1.1
    y_cam = 1.625
    z_cam = 0.723
    r_camx = 70
    r_camy = 0
    r_camz = 145

    # skewed angle y
    x_cam = 1.1
    y_cam = -1.625
    z_cam = 0.723
    r_camx = 70
    r_camy = 0
    r_camz = 35

    # skewed angle * 0.25
    x_cam = .55
    y_cam = 0.8125
    z_cam = 0.3615
    r_camx = 70
    r_camy = 0
    r_camz = 145

    # skewed angle y
    x_cam = 0.55
    y_cam = -0.8125
    z_cam = 0.3615
    r_camx = 70
    r_camy = 0
    r_camz = 35

    '''
    # skewed angle * 0.15
    x_cam = 0.43
    y_cam = 0.634
    z_cam = 0.24
    r_camx = 75
    r_camy = 0
    r_camz = 145

    scene.camera.location.x = box_length * x_cam
    scene.camera.location.y = box_length * y_cam
    scene.camera.location.z = box_length * z_cam

    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = (np.pi/180.0) * r_camx
    scene.camera.rotation_euler[1] = (np.pi/180.0) * r_camy
    scene.camera.rotation_euler[2] = (np.pi/180.0) * r_camz

    # Sets field of view
    #scene.camera.data.angle = 50*(np.pi/180.0)
    #bpy.data.cameras['Camera'].type = 'ORTHO'
    #bpy.data.cameras['Camera'].ortho_scale = 21.0

    # change lamp position
    bpy.data.objects["Lamp"].location = (0,0,5)

    # Remove lighting (for now)
    remove_obj( scene )

    # sets background to be black
    bpy.data.worlds['World'].horizon_color = (0,0,0)

    return scene

# Adds lines to specific data points for eigenvector testing
# Because this is example specific, I will not return anything.
def add_lines(connectome):
    for i in connectome:
        bpy.ops.curve.primitive_bezier_curve_add()
        bpy.context.object.data.splines.active.id_data.name = "bc" + str(i[0])
        #bpy.data.curves["BezierCurve"] = "bc" + str(i[0])
        bpy.data.curves["bc" + str(i[0])].bevel_depth = 0.01
        bpy.data.curves["bc" + str(i[0])].bevel_resolution = 4
        bpy.data.curves["bc" + str(i[0])].fill_mode = 'FULL'
        bpy.data.curves["bc" + str(i[0])].splines[0].bezier_points[0].co = \
            bpy.context.scene.objects[str(i[1])].location
        bpy.data.curves["bc" + str(i[0])].splines[0].bezier_points[0].handle_left_type = 'VECTOR'
        bpy.data.curves["bc" + str(i[0])].splines[0].bezier_points[1].co = \
            bpy.context.scene.objects[str(i[2])].location
        bpy.data.curves["bc" + str(i[0])].splines[0].bezier_points[1].handle_left_type = 'VECTOR'
        color = (1, 1, 1)
        mat = create_new_material("bc" + str(i[0]), color)
        bpy.data.curves["bc" + str(i[0])].materials.append(mat)
        bpy.data.curves["bc" + str(i[0])].splines[0].resolution_u = 1
    
# Function to create list of all possible nearest neighbors
# Reads in resolution of one side of cube of points
def create_connectome(res):
    connectome = []
    count = 0
    pnum = 0
    for i in range(res):
        for j in range(res):
           for k in range(res):
               if (k + 1 < res and pnum + 1 < res * res * res):
                   connectome.append([count, pnum, pnum+1])
                   count = count + 1
               if ((k+(j*res)+res < res*res) and (pnum+res < res*res*res)):
                   connectome.append([count, pnum, pnum + res])
                   count = count + 1
               if (pnum + res * res < res * res * res):
                   connectome.append([count, pnum, pnum + res * res])
                   count = count + 1
               pnum = pnum + 1

    print("length of the connectome is: ", len(connectome))
    return connectome

# Renders movie
def set_render_options(scene, res_x, res_y):

    # Scene resolution
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    scene = bpy.context.scene
    bpy.data.scenes[0].render.image_settings.file_format="PNG"
    #bpy.data.scenes[0].render.filepath = "images/image%.5d" 
    bpy.ops.render.render( write_still=True )
    scene.sequence_editor_create()
    bpy.data.scenes["Scene"].render.resolution_percentage = 100
    bpy.data.scenes["Scene"].render.fps = 60
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'FFMPEG'
    #bpy.data.scenes["Scene"].render.ffmpeg.audio_codec = 'NONE'
    bpy.data.scenes["Scene"].render.ffmpeg.constant_rate_factor='PERC_LOSSLESS'
    bpy.data.scenes["Scene"].render.filepath = 'out.mkv'
    bpy.data.scenes["Scene"].render.use_file_extension = False

def render_movie(scene):
    print("rendering movie")
    bpy.ops.render.render( animation=True ) 

# Function for visualization of slicing cube
def cube_slice(res):
    # Defining the initial cube
    cube = define_tesseract(1, 0.5)

    # defining sphere material for copying
    new_sphere(0.05, 0, 0, 0, 1, 0, 0, "original")

    count = 0
    for i in cube:
        place_duplicates(i.projx, i.projy, i.projz, count, "original");
        count += 1

    bpy.data.objects["original"].hide = True
    bpy.data.objects["original"].hide_render = True

    plane = bpy.ops.mesh.primitive_plane_add(location=(0,0,-0.5),
                                             radius = 1)

    #mat = bpy.data.materials.new("Plane")
    #ob = bpy.context.object
    #me = ob.data

    mat = create_new_material("Plane", (0,0,1))
    mat.use_transparency = True
    mat.alpha = 0.5
    bpy.data.objects["Plane"].data.materials.append(mat)

    connectome = create_connectome(2)
    add_lines(connectome)

    for i in range (0, res):
        # updating plane pos
        mat = bpy.data.objects["Plane"]
        mat.keyframe_insert(data_path="location", \
            frame=(i), index=-1)
        bpy.context.scene.objects["Plane"].location =  \
            (0, 0, -1 + 2*i / (res-1))
        bpy.context.scene.objects["Plane"].keyframe_insert(
            data_path='location', frame=(i))

'''
num = 100
scene = bpy.context.scene
scene = def_scene(10,scene)
remove_obj(scene)
#cube_slice(100)
project_cube(100)
#visualize_fourth_dimension(1, 100, 100)

# Adding in extra function for determinant visualization
bpy.data.scenes["Scene"].frame_end = num
scene.update()
render_movie(scene)
#print (array)
'''
