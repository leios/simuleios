#-------------determinant.py---------------------------------------------------#
#
#             determinant.py
# 
# Purpose: Visualize a simple matrix multiplication
#
#   Notes: This code can be run by using the following command:
#              blender -b -P eigentest.py
#          To show on stream, use:
#              mplayer out.mp4 -vo x11
#          Figure out if we are in the exterior box.
#
#------------------------------------------------------------------------------#

import bpy
import numpy as np
import random

# goes through all the data! Woo!
# Written (in part) by Kramsfasel
def parse_data(num_part=0):
    array = []

    # Set of connections for lines later on
    connectome = create_connectome(2)
    #connectome = []
    #connectome.append([0,0,1])
    #connectome.append([1,2,3])
    #connectome.append([2,4,5])
    #connectome.append([3,6,7])
    offset = 0
    linesInDataSet = 0
    print("importing data from file")
    #input = "../MD/demon/out.dat"
    input = "/tmp/file.dat"
    input = "out.dat"
    num_part_temp = 0
    with open(input, 'r') as data:
        for line in data:
            if line != '\n':
                linesInDataSet +=1
                s = line.split()
                temp = [float(s) for s in line.split()]
                temp[7]=int (s[7])
                temp[6]=int (s[6])
                array.append(temp)
            else:
                if (num_part == 0):
                    num_part=linesInDataSet

    (max_vel, min_vel) = place_spheres(array, num_part, linesInDataSet)
    add_lines(connectome)
    (max_vel, min_vel) = place_spheres(array, num_part, linesInDataSet)
    add_lines(connectome)

    numberOfFrames = int (linesInDataSet / num_part) 

    print ("found " + str(numberOfFrames) + " and " + str(num_part) + " particles in first frame")   

    for linesInDataSet in range(2, numberOfFrames+1):
        if (linesInDataSet%100==0):
            print ("at frame "+str(linesInDataSet)+" of "+str(numberOfFrames))
        move_spheres(array, num_part, linesInDataSet, max_vel, min_vel)
        move_lines(connectome, linesInDataSet)
    return numberOfFrames

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

# places new sphere at given location
def new_sphere_MC(diam, x, y, z, r, g, b, id):
    temp_sphere = bpy.ops.mesh.primitive_uv_sphere_add(segments = 8, 
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
def cage_set(Box_length, make_frame):
    ccube = bpy.ops.mesh.primitive_cube_add(location=(0,0,0),
                                            radius = Box_length / 2)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects['Cube.001'].hide = True
    bpy.data.objects['Cube.001'].hide_render = True
    bpy.data.objects['Cube.001'].keyframe_insert("hide_render", 
                                                 frame=make_frame-1)
    bpy.data.objects['Cube.001'].keyframe_insert("hide", 
                                                 frame=make_frame-1)
    bpy.data.objects['Cube.001'].hide = False
    bpy.data.objects['Cube.001'].hide_render = False
    bpy.data.objects['Cube.001'].keyframe_insert("hide_render", 
                                                 frame=make_frame)
    bpy.data.objects['Cube.001'].keyframe_insert("hide", 
                                                 frame=make_frame)

    ob = bpy.context.object
    me = ob.data
    mat = create_cage('MaterialCage')
    me.materials.append(mat)
    return ccube

# Removes objects in scene
def remove_obj( scene ):
    for ob in scene.objects: 
        if ob.name !='Camera' and ob.name != 'Lamp':
            scene.objects.unlink( ob )

#defining our scene
def def_scene(box_length, bgcolor):

    # Camera stuff
   
    '''
    x_cam = 2.2
    y_cam = 2.75
    z_cam = 1.43
    r_camx = 70
    r_camy = 0
    r_camz = 145

    x_cam = 0
    y_cam = 0.5
    z_cam = 4
    r_camx = 0
    r_camy = 0
    r_camz = 0

    '''
    x_cam = 1.1
    y_cam = 1.625
    z_cam = 0.723
    r_camx = 70
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

    # Scene resolution
    scene.render.resolution_x = 1366*3
    scene.render.resolution_y = 1024*3
    #scene.render.resolution_y = 768

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
def render_movie(scene):
    scene = bpy.context.scene
    bpy.data.scenes[0].render.image_settings.file_format="PNG"
    #bpy.data.scenes[0].render.filepath = "images/image%.5d" %iteration
    bpy.ops.render.render( write_still=True )
    print("rendering movie")
    scene.sequence_editor_create()
    bpy.data.scenes["Scene"].render.fps = 30
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'FFMPEG'
    #bpy.data.scenes["Scene"].render.ffmpeg.video_bitrate = 24300
    bpy.data.scenes["Scene"].render.ffmpeg.format = 'MPEG4'
    bpy.data.scenes["Scene"].render.ffmpeg.audio_codec = 'NONE'
    bpy.data.scenes["Scene"].render.ffmpeg.minrate = 0
    bpy.data.scenes["Scene"].render.ffmpeg.maxrate = 30000
    bpy.data.scenes["Scene"].render.ffmpeg.codec = 'MPEG4'
    bpy.data.scenes["Scene"].render.filepath = 'out.mp4'
    bpy.data.scenes["Scene"].render.use_file_extension = False
    bpy.ops.render.render( animation=True ) 

# Hiding original cube until a particular time
def hide_interior(make_frame):
    # hiding the particles
    for i in range(8):
        name = "%d.001" % (i)
        print(name)
        bpy.data.objects[name].hide = True
        bpy.data.objects[name].hide_render = True
        bpy.data.objects[name].keyframe_insert("hide_render", 
                                               frame=make_frame-1)
        bpy.data.objects[name].keyframe_insert("hide", 
                                               frame=make_frame-1)
        bpy.data.objects[name].hide = False
        bpy.data.objects[name].hide_render = False
        bpy.data.objects[name].keyframe_insert("hide_render", 
                                               frame=make_frame)
        bpy.data.objects[name].keyframe_insert("hide", 
                                               frame=make_frame)

    # hiding of the lines
    for i in range(12):
        if i == 0:
            name = "BezierCurve"
        else:
            name = "BezierCurve.%03d" % (i)
        print(name)
        bpy.data.objects[name].hide = True
        bpy.data.objects[name].hide_render = True
        bpy.data.objects[name].keyframe_insert("hide_render", 
                                               frame=make_frame-1)
        bpy.data.objects[name].keyframe_insert("hide", 
                                               frame=make_frame-1)
        bpy.data.objects[name].hide = False
        bpy.data.objects[name].hide_render = False
        bpy.data.objects[name].keyframe_insert("hide_render", 
                                               frame=make_frame)
        bpy.data.objects[name].keyframe_insert("hide", 
                                               frame=make_frame)

# Monte carlo for this special case of two internal cubes
def monte_carlo(framenum, resolution, box_length):
    tot_count = 0
    in_count = 0
    ex_count = 0
    create_MCspheres(0.05)
    for j in range(1):
        for i in range(resolution):
            MCid = "MCB"
            tot_count += 1
            count = i + j * resolution
            x = random.random() * box_length - box_length * 0.5
            y = random.random() * box_length - box_length * 0.5
            z = random.random() * box_length - box_length * 0.5
    
            (in_box, MCid) = in_exterior(x, y, z, MCid, in_count)
            ex_count += in_box
            (in_box, MCid) = in_interior(x, y, z, MCid, ex_count)
            in_count += in_box
    
            # MC points are above 100...
            place_duplicates(x, y, z, "MC%d" % count, MCid)
    
            # Adding in the appropriate Keyframes
    
            bpy.data.objects["MC%d" % count].hide = True
            bpy.data.objects["MC%d" % count].hide_render = True
            bpy.data.objects["MC%d" % count].keyframe_insert("hide_render", 
                                                         frame=framenum-1)
            bpy.data.objects["MC%d" % count].keyframe_insert("hide", 
                                                         frame=framenum-1)
            bpy.data.objects["MC%d" % count].hide = False
            bpy.data.objects["MC%d" % count].hide_render = False
            bpy.data.objects["MC%d" % count].keyframe_insert("hide_render", 
                                                         frame=framenum)
            bpy.data.objects["MC%d" % count].keyframe_insert("hide", 
                                                         frame=framenum)


        framenum += 1
        print(i)
    in_vol = in_count / tot_count * box_length
    ex_vol = ex_count / tot_count * box_length
    vol_ratio = ex_vol / in_vol
    with open('volumes.dat','w') as outfile:
        outfile.write("{} {} {}".format(in_vol, ex_vol, vol_ratio)) 
 
    return framenum

def in_interior(x, y, z, MCid, in_count):
    in_box = 0
    if x > -1 and x <= 1 and y > -1 and y <= 1 and z > -1 and z <= 1:
        in_box = 1
        MCid = "MCR"
    return (in_box, MCid)

def in_exterior(x, y, z, MCid, ex_count):
    in_box = 0
    # Defining the transformation matrix and inverse
    matrix = np.matrix('1 2 0; 2 1 0; 0 0 -3')
    inverse = np.linalg.inv(matrix)
    rand_vector = np.array([x, y, z])

    # Now transforming back and checking against original cube
    point = rand_vector * inverse

    if point[0,0] > -1 and point[0,0] <= 1 \
       and point[0,1] > -1 and point[0,1] <= 1 \
       and point[0,2] > -1 and point[0,2] <= 1:
        MCid = "MCY"
        in_box = 1
    return (in_box, MCid)

def create_MCspheres(diam):
    new_sphere_MC(diam, 0, 0, 0, 1, 0, 0, "MCR")
    bpy.data.materials["MCR"].use_transparency = True
    bpy.data.materials["MCR"].alpha = 0.25
    bpy.data.objects["MCR"].hide = True
    bpy.data.objects["MCR"].hide_render = True

    new_sphere_MC(diam, 0, 0, 0, 1, 1, 0, "MCY")
    bpy.data.materials["MCY"].use_transparency = True
    bpy.data.materials["MCY"].alpha = 0.25
    bpy.data.objects["MCY"].hide = True
    bpy.data.objects["MCY"].hide_render = True

    new_sphere_MC(diam, 0, 0, 0, 0, 0, 1, "MCB")
    bpy.data.materials["MCB"].use_transparency = True
    bpy.data.materials["MCB"].alpha = 0.25
    bpy.data.objects["MCB"].hide = True
    bpy.data.objects["MCB"].hide_render = True

def vis_determinant(box_length, num):
    hide_interior(num)
    num += 10

    cage_set(box_length, num)
    num += 10

    num = monte_carlo(num, 1000, box_length)

    return num

scene = bpy.context.scene
scene = def_scene(10,scene)
remove_obj(scene)
num = parse_data()

# Adding in extra function for determinant visualization
num = vis_determinant(6.0, num)
bpy.data.scenes["Scene"].frame_end = num
scene.update()
#render_movie(scene)
#print (array)
