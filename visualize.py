import bpy

import sys, os

from mathutils import Matrix, Vector

import math

import numpy as np

import json

import time

# This is required to print output to blender console (needed if starting blender not from command line)

#def print(*data):

#    for window in bpy.context.window_manager.windows:

#        screen = window.screen

#        for area in screen.areas:

#            if area.type == 'CONSOLE':

#                override = {'window': window, 'screen': screen, 'area': area}

#                bpy.ops.console.scrollback_append(override, text=str(" ".join([str(x) for x in data])), type="OUTPUT")      

##                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")       

path_to_files = 'sim_img_cache'

path = bpy.path.abspath('//') + path_to_files

# Getting the list of directories
dir = os.listdir(path)

# Checking if the list is empty or not
if len(dir) == 0:
    print("Empty directory")
else:
    raise ValueError('Directory is not empty!')

scene = bpy.context.scene

# Form video
camera = bpy.data.objects['Camera']

iter = 0
while True:
    path_to_pose = path + f'/{iter}.json'
    path_to_img = path + f'/{iter}.png'
    if os.path.exists(path_to_pose) is True:
        with open(path_to_pose,"r") as f:
            pose = json.load(f)
        camera.matrix_world = Matrix(pose)

        # save image from camera
        bpy.context.scene.render.filepath = path_to_img
        bpy.context.scene.render.resolution_x = 800
        bpy.context.scene.render.resolution_y = 800
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        bpy.ops.render.render(write_still = True)

        iter += 1

    time.sleep(0.01)
print("--------------------    DONE WITH BLENDER SCRIPT    --------------------")