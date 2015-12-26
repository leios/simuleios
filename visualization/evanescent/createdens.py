#-----------createdens.py------------------------------------------------------#
#
#           Create Density plot for 3d FDTD
#
# Purpose: Create cool visualizations for FDTD in 3d
#
#   Notes: This is a dummy file until tomorrow.
#          To run this script, use:
#              blender -b -P createdens.py
#                  -b runs without GUI
#                  -P executes python script
#          Startup file has no cube or lighting. This can be set by initilially
#              removing the lighting and cube and then clicking the "Save 
#              Startup File" button; however, we remove the objects too.
#
#------------------------------------------------------------------------------#

import bpy
import numpy as np
import struct

# Files and data and such
voxelfile = "3Dsample.raw"
input = "3Devanescent.dat"
infile = open(input,'r')
outfile = open(voxelfile,'wb')
vdata = np.genfromtxt(input)

# Removes objects in scene
def remove_obj( scene ):
    for ob in scene.objects: 
        if ob.name !='Camera':
            scene.objects.unlink( ob )

# Define Scene
def def_scene(box_length, res_stand, xres, yres, zres):

    # Camera stuff
    '''
    x_cam = 2.2
    y_cam = 2.75
    z_cam = 1.45

    x_cam = 0
    y_cam = 0.5
    z_cam = 4
    '''

    x_cam = 3.0
    y_cam = 3.2
    z_cam = 2.0

    scene = bpy.context.scene

    remove_obj(scene)

    # Defining dummy point to point camera at
    bpy.ops.object.add(type='EMPTY', location=(box_length * xres * 0.5 / res_stand, box_length * yres * 0.5 / res_stand, box_length * zres * 0.5 / res_stand))

    # This is not working. Cannot figure out how to select camera.
    context = bpy.context

    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.context.scene.objects.active = bpy.context.scene.objects["Camera"]
    ob = bpy.data.objects['Camera']
    bpy.ops.object.constraint_add(type="TRACK_TO")
    target = bpy.data.objects.get('Empty', False)
    ob.constraints['Track To'].target=target

    ob.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'
    ob.constraints['Track To'].up_axis = 'UP_Y'

    scene.camera.location.x = box_length * x_cam * xres/res_stand
    scene.camera.location.y = box_length * y_cam * yres/res_stand
    scene.camera.location.z = box_length * z_cam * zres/res_stand

    # Sets field of view
    scene.camera.data.angle = 50*(np.pi/180.0)
    #bpy.data.cameras['Camera'].type = 'ORTHO'
    bpy.data.cameras['Camera'].ortho_scale = 21.0

    # Scene resolution
    scene.render.resolution_x = 1366*2
    scene.render.resolution_y = 768*2

    # set number of cores used
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 8

    # sets background to be black
    bpy.data.worlds['World'].horizon_color = (0,0,0)

    return scene

# Create Cube
def createcube(box_length, res_stand, xres, yres, zres, step_size, dens_scale, voxelfile, color_num):
    cube = bpy.ops.mesh.primitive_cube_add(location=((box_length * xres / (2 * res_stand)), (box_length * yres / (2 * res_stand)), (box_length *zres / (2 * res_stand))), radius = box_length / 2)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.scale=((xres/res_stand,yres/res_stand,zres/res_stand))
    me = ob.data
    mat = createVolume('MaterialVolume', xres, yres, zres, step_size, dens_scale
, voxelfile, color_num)
    me.materials.append(mat)
    bpy.ops.object.modifier_add(type="WIREFRAME")
    cageMat = createCage("wire")
    me.materials.append(cageMat)
    bpy.data.objects["Cube"].modifiers["Wireframe"].material_offset = 1
    bpy.data.objects["Cube"].modifiers["Wireframe"].use_replace = False
    return cube

# Create Material
def createVolume (passedName, xres, yres, zres, step_size, dens_scale, voxelfile, color_num):
    volMat = bpy.data.materials.new(passedName)
    volMat.type = 'VOLUME'
    volMat.volume.density = 0.0
    volMat.volume.step_method = 'CONSTANT'
    volMat.volume.step_size = step_size
    volMat.volume.depth_threshold = 0.010
    volMat.volume.density_scale = dens_scale
    matTex = volMat.texture_slots.add()
    voxTex = bpy.data.textures.new('VoxelData', type = 'VOXEL_DATA')
    voxTex.voxel_data.file_format = 'RAW_8BIT'
    voxTex.use_color_ramp = True
    voxTex.color_ramp.color_mode = "RGB"
    ramp = voxTex.color_ramp

    values = [(0.0,(0,0,1,0)), (0.5,(1,0,1,0.3)), (1.0, (1,0,0,1))]

    for n,value in enumerate(values):
        ramp.elements.new((n+1)*0.2)
        elt = ramp.elements[n]
        (pos, color) = value
        elt.position = pos
        elt.color = color
    voxTex.voxel_data.filepath = voxelfile
    voxTex.voxel_data.resolution = (xres, yres, zres)
    matTex.texture = voxTex
    matTex.use_map_to_bounds = True
    matTex.texture_coords = 'ORCO'
    matTex.use_map_color_diffuse = True 
    matTex.use_map_emission = True 
    matTex.emission_factor = 1
    matTex.emission_color_factor = 1
    matTex.use_map_density = True 
    matTex.density_factor = 1
    return volMat

def createCage(passedName):
    cageMat = bpy.data.materials.new(passedName)
    cageMat.use_shadeless = True
    return cageMat

# Render Scene into image
def render_img(filename):
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.ops.render.render( write_still=True )

# Render_Movie
def render_movie():
    scene = bpy.context.scene
    #bpy.data.scenes[0].render.image_settings.file_format="PNG"
    bpy.ops.render.render( write_still=True )
    print("rendering movie")
    scene.sequence_editor_create()
    bpy.data.scenes["Scene"].render.fps = .1
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'FFMPEG'
    bpy.data.scenes["Scene"].render.ffmpeg.video_bitrate = 1000
    bpy.data.scenes["Scene"].render.ffmpeg.format = 'MPEG4'
    bpy.data.scenes["Scene"].render.ffmpeg.audio_codec = 'NONE'
    bpy.data.scenes["Scene"].render.ffmpeg.minrate = 0
    bpy.data.scenes["Scene"].render.ffmpeg.maxrate = 1500
    bpy.data.scenes["Scene"].render.ffmpeg.codec = 'H264'
    bpy.data.scenes["Scene"].render.filepath = 'out.mp4'
    bpy.data.scenes["Scene"].render.use_file_extension = False
    bpy.data.scenes["Scene"].frame_end = 40
    bpy.ops.render.render( animation=True ) 

# function to write data to .raw file for blender
# note, the density muct be an integer between 0 and 255
def voxel_gen(vdata, outfile, ii):
    print("generating voxel data.")
    for i in range(0,ii):
        #print(i)
        outfile.write(struct.pack('B', abs(int(vdata[i]))))
    outfile.flush()
    outfile.close()


#------------------------------------------------------------------------------#
# MAIN
#------------------------------------------------------------------------------#

voxel_gen(vdata, outfile, len(vdata))
def_scene(5,64,128,64,64)
createcube(5,64,128,64,64,0.1,0.5, voxelfile, 1)
#render_movie()
render_img("image.png")
