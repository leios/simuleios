#-----------createdata.py------------------------------------------------------#
#
#           Create Density plot for 3d FDTD
#
# Purpose: Takes data from 3Devanescent.cpp and turns it into a blender readible
#          format. 
#
#------------------------------------------------------------------------------#

#import bpy
import numpy as np
import struct

# Files and data and such
infile = open("sample_data.dat",'r')
outfile = open("raw_plot.raw",'wb')
vdata = np.genfromtxt("sample_data.dat")

# function to write data to .raw file for blender
# note, the density muct be an integer between 0 and 255
def voxel_gen(vdata, vfile, ii):
    for i in range(0,ii):
        print(i)
        vfile.write(struct.pack('B', abs(int(vdata[i][3]))))
    vfile.flush()
    vfile.close()

#------------------------------------------------------------------------------#
# MAIN
#------------------------------------------------------------------------------#
voxel_gen(vdata, outfile, len(vdata))
