#-------------createsample.py--------------------------------------------------#
#
#             create sample.py
#
# Purpose: To create sample blender data for testing voxel stuff with.
#
#------------------------------------------------------------------------------#


#import bpy
import numpy as np
import struct

# Files and data and such
vfile = open("sample.raw",'wb')

n = 64
vdata = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]

# creating initial gradient for viewing. 0 to 255 at x,y,z = box_length
for i in range(64):
    for j in range(64):
        for k in range(64):
            vdata[i][j][k] = (0.33 * (i/64) + 0.33 * (j/64) + 0.33 * (k/64))*255

# function to write data to .raw file for blender
# note, the density muct be an integer between 0 and 255
def voxel_gen(vdata, vfile, ii):
    for i in range(ii):
        for j in range(ii):
            for k in range(ii):

                vfile.write(struct.pack('B', abs(int(vdata[i][j][k]))))
    vfile.flush()
    vfile.close()

voxel_gen(vdata, vfile, 64)
