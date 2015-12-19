#-----------createdens.py------------------------------------------------------#
#
#           Create Density plot for 3d FDTD
#
# Purpose: Create cool visualizations for FDTD in 3d
#
#   Notes: This is a dummy file until tomorrow.
#
#     Add: Color ramp
#------------------------------------------------------------------------------#

import bpy
import numpy as np
import struct

# Files and data and such
infile = open("sample_data.dat",'r')
vfile = open("raw_plot.raw",'wb')
voxelfile = "sample.raw"
vdata = np.genfromtxt("sample_data.dat")

# function to write data to .raw file for blender
# note, the density muct be an integer between 0 and 255
def voxel_gen(vdata, vfile, ii):
    for i in range(0,ii):
        vfile.write(struct.pack('B', abs(int(vdata[i][3]))))
    vfile.flush()
    vfile.close()

# Define Scene
def def_scene(box_length):
    # Camera stuff
    x_cam = 2.2
    y_cam = 2.75
    z_cam = 1.43
    r_camx = 70
    r_camy = 0
    r_camz = 145

    '''
    x_cam = 0
    y_cam = 0.5
    z_cam = 4
    r_camx = 0
    r_camy = 0
    r_camz = 0
    '''
    

    scene = bpy.context.scene

    scene.camera.location.x = box_length * x_cam
    scene.camera.location.y = box_length * y_cam
    scene.camera.location.z = box_length * z_cam

    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = (np.pi/180.0) * r_camx
    scene.camera.rotation_euler[1] = (np.pi/180.0) * r_camy
    scene.camera.rotation_euler[2] = (np.pi/180.0) * r_camz

    # Sets field of view
    scene.camera.data.angle = 50*(np.pi/180.0)
    bpy.data.cameras['Camera'].type = 'ORTHO'
    bpy.data.cameras['Camera'].ortho_scale = 21.0

    # Scene resolution
    scene.render.resolution_x = 1366*2
    scene.render.resolution_y = 768*2
    scene.render.threads = 8

    # sets background to be black
    bpy.data.worlds['World'].horizon_color = (0,0,0)

    return scene

# Create Cube
def createcube(box_length, xres, yres, zres, step_size, dens_scale, voxelfile, color_num):
    cube = bpy.ops.mesh.primitive_cube_add(location=((box_length / 2), (box_length / 2), (box_length / 2)), radius = box_length / 2)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    #ob.scale=(0.5,0.5,0.5)
    me = ob.data
    mat = createVolume('MaterialVolume', xres, yres, zres, step_size, dens_scale
, voxelfile, color_num)
    me.materials.append(mat)
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

    '''
    values = det_color(color_num)

    for n,value in enumerate(values):
        ramp.elements.new((n+1)*0.2)
        elt = ramp.elements[n]
        (pos, color) = value
        elt.position = pos
        elt.color = color
    '''
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

# Render Scene into image
def render_img(filename):
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.ops.render.render( write_still=True )


#------------------------------------------------------------------------------#
# MAIN
#------------------------------------------------------------------------------#

def_scene(5)
createcube(5,64,64,64,0.1,0.5, voxelfile, 1)
render_img("check.png")
