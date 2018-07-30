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

num = 10
scene = bpy.context.scene
scene = def_scene(10,scene)
remove_obj(scene)

define_axes(1)

new_plane(0,0,0, 0,0,0,         0,0,1,0.5, 1,"plane1")
new_plane(0,0,0, 0,0.5*np.pi,0, 1,0,0,0.5, 1,"plane2")
new_plane(0,0,0, 0.5*np.pi,0,0, 0,1,0,0.5, 1,"plane3")

# Adding in extra function for determinant visualization
bpy.data.scenes["Scene"].frame_end = num
scene.update()
render_movie(scene)
