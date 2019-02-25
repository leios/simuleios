#-------------gravity.py-------------------------------------------------------#
#
# Purpose: this file intends to provide a quick demonstration of how to use
#          blender python to make animations. In particular, we plan to make 
#          an animation of a ball dropping to the ground and one of a 
#          solar system
#
#   Notes: To create the animation, use the following command:
#              blender -b -P gravity.py
#          This should create a video for you to play. You can also open the 
#          animation in blender by running:
#              blender -P gravity.py
#            
#------------------------------------------------------------------------------#

import bpy
import numpy as np

class particle:
    x = 0
    y = 0
    z = 0
    r = 0
    g = 0
    b = 0
    size = 1.0
    def __init__(self, xloc, yloc, zloc, rcolor, gcolor, bcolor, s):
        self.x = xloc
        self.y = yloc
        self.z = zloc
        self.r = rcolor
        self.g = gcolor
        self.b = bcolor
        self.size = s

def new_sphere(p, ob_id):
    sphere = bpy.ops.mesh.primitive_uv_sphere_add(segments = 32,
                                                  ring_count = 16,
                                                  size = p.size,
                                                  location = (p.x, p.y, p.z),
                                                  rotation = (0, 0, 0))
    ob = bpy.context.active_object
    ob.name = str(ob_id)
    mat = create_new_material(ob.name, (p.r, p.g, p.b))
    ob.data.materials.append(mat)
    return sphere

def clear_scene(scene):
    for ob in scene.objects:
        if ob.name != 'Camera' and ob.name != 'Lamp':
            scene.objects.unlink(ob)

def create_scene(scale, scene):
    # Creating multiplicative factors to scale by
    scene.camera.location.x = scale * 0.43
    scene.camera.location.y = scale * 0.634
    scene.camera.location.z = scale * 0.24
    scene.camera.rotation_mode = 'XYZ'
    scene.camera.rotation_euler[0] = (np.pi/180.0) * 75
    scene.camera.rotation_euler[1] = (np.pi/180.0) * 0
    scene.camera.rotation_euler[2] = (np.pi/180.0) * 145

    # change lamp position
    bpy.data.objects["Lamp"].location = (0,0,5)

    # Set background to be black
    bpy.data.worlds['World'].horizon_color = (0,0,0)

    return scene

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

# Renders movie
def render_movie(scene):
    print("rendering movie")
    bpy.ops.render.render( animation=True )


