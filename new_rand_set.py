#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:03:26 2020

@author: milly
"""

import numpy as np
from tomophantom import TomoP2D 
from tomophantom.TomoP2D import Objects2D
import astra
import os

num_gen = 1000

#make this automatic
trainset = 3
min_obs = 5
max_obs = 30

sinogram_folder = "sinograms"
phantom_folder = "phantoms"

trainset_list = list(filter(lambda x: "sinograms" in x, os.listdir(os.getcwd())))
trainset_list = list(map(lambda x: int(x.split("_")[-1]), trainset_list))
while trainset in trainset_list:
    trainset += 1
    
root_dir = os.getcwd()

sinogram_folder = os.path.join(root_dir, sinogram_folder + "_{}_{}_{}".format(min_obs, max_obs, trainset))
phantoms_folder = os.path.join(root_dir, phantom_folder + "_{}_{}_{}".format(min_obs, max_obs, trainset))
os.mkdir(sinogram_folder)
os.mkdir(phantoms_folder)

sino_name = "sino_{}"
phantom_name = "ground_{}"

#keeps the generated objects within a circle with a diameter 90% of the 
width = 0.9

num_objects = np.random.randint(min_obs, max_obs)
ob_list = np.empty(num_objects, dtype=dict)

for i in range(num_gen):
    print(i)
    for ob_index in range(num_objects):
    
      density = np.random.rand()
      shape = [Objects2D.ELLIPSE, Objects2D.RECTANGLE][np.random.randint(0,2)]
      
      if shape == Objects2D.ELLIPSE:
        R = width
        x = np.random.rand() * 2 - 1
        y = np.random.rand() * 2 - 1
        while x**2 + y**2 > R ** 2:
          x = np.random.rand() * 2 - 1
          y = np.random.rand() * 2 - 1
        max_length = R - (x**2 + y**2) ** 0.5
        long_length = np.random.rand() * max_length
        short_length = np.random.rand() * long_length
      
      if shape == Objects2D.RECTANGLE:
        R = width/2
        x = np.random.rand() * 2
        y = np.random.rand() * 2
        while x**2 + y**2 > R ** 2:
          x = np.random.rand() * 2
          y = np.random.rand() * 2
        max_length = R - (x**2 + y**2) ** 0.5
        max_length *= 4/(2**0.5)
        long_length = np.random.rand() * max_length
        short_length = np.random.rand() * long_length
    
      rot = np.random.randint(0, 360)
      ob = {'Obj': shape,
            'C0' : density,
            'x0' : x,
            'y0' : y,
            'a'  : long_length,
            'b'  : short_length,
            'phi': rot}
    
      ob_list[ob_index] = ob
    
    #make these choices
    phantom = TomoP2D.Object(1499, ob_list)
    
    vol_geom = astra.creators.create_vol_geom(1499, 1499, -256, 256, -256, 256)
    proj_geom = astra.creators.create_proj_geom('parallel',1, 
                                           512, np.linspace(0, np.pi, 512, False))
    proj_id = astra.create_projector("cuda",proj_geom,vol_geom)
    vol_geom_rec = astra.create_vol_geom(512,512)
    sino_id, sinogram = astra.create_sino(phantom,proj_id, gpuIndex=1)
    
    np.save(os.path.join(phantoms_folder, phantom_name.format(i)), phantom)
    np.save(os.path.join(sinogram_folder, sino_name.format(i)), sinogram)
