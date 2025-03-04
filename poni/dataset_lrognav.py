import gc
import cv2
import os 
import bz2
import math
import json
import tqdm
import h5py
import glob
import torch
import random
import numpy as np

np.set_printoptions(threshold=np.inf)

import os.path as osp
import _pickle as cPickle
import skimage.morphology as skmp
import calendar;
import matplotlib.pyplot as plt
import time;

dir="/home/aae14859ln/Sun/PONI/tmp_img/"

global tform_trans, tform_rot
tform_trans = None
tform_rot = None


from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
from poni.geometry import (
    spatial_transform_map,
    crop_map,
    get_frontiers_np,
)
from poni.constants import (
    SPLIT_SCENES,
    OBJECT_CATEGORIES,
    INV_OBJECT_CATEGORY_MAP,
    NUM_OBJECT_CATEGORIES,
    # General constants
    CAT_OFFSET,
    FLOOR_ID,
    # Coloring
    d3_40_colors_rgb,
    gibson_palette,
    gibson_palette_room,
)
from poni.fmm_planner import FMMPlanner
from einops import asnumpy, repeat
from matplotlib import font_manager

MIN_OBJECTS_THRESH = 4
EPS = 1e-10


# # cancel all the print info
# def print(*args, **kwargs):
#     pass




GIBSON_ROOM_CATEGORIES = [
    "out-of-bounds",
        "floor",
        "wall",
        "bathroom",
        "bedroom",
        "childs_room",
        "closet",
        "corridor",
        "dining_room",
        "empty_room",
        "exercise_room",
        "garage",
        "home_office",
        "kitchen",
        "living_room",
        "lobby",
        "pantry_room",
        "playroom",
        "staircase",
        "storage_room",
        "television_room",
        "utility_room",]

GIBSON_OBJECT_CATEGORIES = [
        "out-of-bounds",
        "floor",
        "wall",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv",
        "dining-table",
        "oven",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "cup",
        "bottle",
    ]

global obj_room_sc
obj_room_sc=  np.zeros((15,19))
obj_room_sc_po = np.zeros((15,19))
obj_room_sc_ne = np.zeros((15,19))
# obj_room_sc_comb = np.zeros((15,19))

# obj_room_sc[0,:] = [0.1, 0.9, 0.8, 0.1, 0.2, 1.0, 0.0, 0.5, 0.2, 0.9, 0.7, 1.0, 0.8, 0.2, 0.7, 0.1, 0.2, 1.0, 0.2]
# obj_room_sc[1,:] = [0.0, 0.4, 0.3, 0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.5, 0.1, 1.0, 0.6, 0.0, 0.7, 0.0, 0.1, 0.9, 0.1]
# obj_room_sc[2,:] = [0.2, 0.3, 0.2, 0.0, 0.4, 0.5, 0.0, 0.3, 0.1, 0.6, 0.4, 0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.7, 0.2]
# obj_room_sc[3,:] = [0.0, 1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0]
# obj_room_sc[4,:] = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
# obj_room_sc[5,:] = [0.1, 0.8, 0.6, 0.0, 0.0, 0.2, 0.0, 0.4, 0.1, 0.7, 0.4, 0.9, 0.3, 0.0, 0.7, 0.0, 0.0, 1.0, 0.1]
# obj_room_sc[6,:] = [0.0, 0.1, 0.2, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.3, 0.7, 0.6, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0]
# obj_room_sc[7,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2]
# obj_room_sc[8,:] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8]
# obj_room_sc[9,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.2]
# obj_room_sc[10,:] = [0.1, 0.7, 0.6, 0.2, 0.1, 0.2, 0.0, 0.1, 0.0, 0.9, 0.1, 0.8, 0.3, 0.0, 0.5, 0.1, 0.4, 0.7, 0.0]
# obj_room_sc[11,:] = [0.2, 0.9, 0.8, 0.1, 0.3, 0.7, 0.0, 0.4, 0.1, 0.9, 0.8, 0.9, 0.7, 0.1, 0.6, 0.2, 0.1, 0.8, 0.1]
# obj_room_sc[12,:] = [0.1, 0.7, 0.5, 0.1, 0.2, 0.9, 0.0, 0.2, 0.1, 0.6, 0.4, 0.9, 0.8, 0.1, 0.5, 0.2, 0.1, 0.7, 0.1]
# obj_room_sc[13,:] = [0.2, 0.4, 0.3, 0.2, 0.1, 0.9, 0.0, 0.2, 0.1, 0.7, 1.0, 0.8, 0.3, 0.6, 0.4, 0.1, 0.2, 0.5, 0.2]
# obj_room_sc[14,:] = [0.8, 0.3, 0.2, 0.4, 0.1, 0.7, 0.0, 0.3, 0.5, 0.4, 1.0, 0.6, 0.2, 0.9, 0.3, 0.1, 0.5, 0.4, 0.6]

obj_room_sc_po[0,:] = [0.1,0.6,0.6,0.1,0.2,0.95,0.3,0.2,0.2,0.9,0.8,0.9,0.7,0.05,0.5,0.05,0.2,0.95,0.1 ]
obj_room_sc_ne[0,:] = [0.9,0.2,0.3,0.8,0.7,0.1,0.9,0.6,0.7,0.1,0.3,0.1,0.5,0.9,0.4,0.9,0.8,0.1,0.8] 
# obj_room_sc[1,:] = [0.0, 0.4, 0.3, 0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.5, 0.1, 1.0, 0.6, 0.0, 0.7, 0.0, 0.1, 0.9, 0.1]
obj_room_sc_po[1,:] = [0.0,0.3,0.2,0.0,0.1,0.1,0.5,0.1,0.1,0.4,0.1,0.95,0.6,0.0,0.7,0.0,0.1,0.95,0.0]
obj_room_sc_ne[1,:] = [0.95,0.4,0.5,0.95,0.9,0.7,0.9,0.8,0.85,0.3,0.6,0.1,0.7,0.95,0.4,0.95,0.9,0.1,0.85]
# obj_room_sc[2,:] = [0.2, 0.3, 0.2, 0.0, 0.4, 0.5, 0.0, 0.3, 0.1, 0.6, 0.4, 0.9, 0.8, 0.1, 0.2, 0.3, 0.1, 0.7, 0.2]
obj_room_sc_po[2,:] = [0.5,0.6,0.4,0.1,0.4,0.7,0.2,0.3,0.1,0.7,0.8,0.9,0.6,0.1,0.4,0.3,0.2,0.7,0.2 ]
obj_room_sc_ne[2,:] = [0.7,0.4,0.4,0.8,0.7,0.5,0.9,0.5,0.8,0.4,0.4,0.3,0.6,0.9,0.4,0.8,0.7,0.3,0.8]
# obj_room_sc[3,:] = [0.0, 1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0]
obj_room_sc_po[3,:] = [0.0,0.95,0.8,0.0,0.0,0.0,0.3,0.0,0.0,0.1,0.0,0.1,0.0,0.0,0.1,0.0,0.1,0.1,0.0]
obj_room_sc_ne[3,:] = [0.95,0.1,0.3,0.95,0.95,0.9,0.6,0.9,0.9,0.8,0.95,0.7,0.95,0.95,0.5,0.95,0.8,0.8,0.95] 
# obj_room_sc[4,:] = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
obj_room_sc_po[4,:] = [0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
obj_room_sc_ne[4,:] = [0.1,0.95,0.95,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.95,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99] 
# obj_room_sc[5,:] = [0.1, 0.8, 0.6, 0.0, 0.0, 0.2, 0.0, 0.4, 0.1, 0.7, 0.4, 0.9, 0.3, 0.0, 0.7, 0.0, 0.0, 1.0, 0.1]
obj_room_sc_po[5,:] = [0.1,0.7,0.5,0.0,0.1,0.2,0.3,0.3,0.2,0.6,0.3,0.9,0.4,0.1,0.6,0.0,0.2,0.95,0.1 ]
obj_room_sc_ne[5,:] = [0.95,0.3,0.4,0.99,0.95,0.7,0.8,0.6,0.85,0.4,0.7,0.1,0.8,0.95,0.5,0.95,0.8,0.1,0.9] 
# obj_room_sc[6,:] = [0.0, 0.1, 0.2, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.3, 0.7, 0.6, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0]
obj_room_sc_po[6,:] = [0.0,0.1,0.1,0.0,0.2,0.95,0.4,0.1,0.1,0.2,0.3,0.2,0.3,0.1,0.2,0.0,0.1,0.2,0.0]
obj_room_sc_ne[6,:] = [0.99,0.95,0.9,0.99,0.95,0.1,0.8,0.95,0.95,0.9,0.7,0.8,0.9,0.95,0.9,0.99,0.9,0.85,0.95]      
# obj_room_sc[7,:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2]
obj_room_sc_po[7,:] = [0.0,0.0,0.0,0.0,0.0,0.1,0.1,0.0,0.0,0.0,0.95,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]
obj_room_sc_ne[7,:] = [0.99,0.99,0.99,0.99,0.99,0.95,0.95,0.98,0.98,0.97,0.2,0.98,0.98,0.99,0.99,0.99,0.95,0.97,0.9]      
# obj_room_sc[8,:] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8]
obj_room_sc_po[8,:] = [0.95,0.1,0.1,0.0,0.1,0.2,0.1,0.1,0.2,0.2,0.95,0.1,0.2,0.2,0.1,0.0,0.1,0.1,0.5]
obj_room_sc_ne[8,:] = [0.1,0.8,0.8,0.95,0.9,0.7,0.9,0.85,0.85,0.75,0.3,0.85,0.85,0.9,0.8,0.95,0.85,0.85,0.7]      
# obj_room_sc[9,:] =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.2]
obj_room_sc_po[9,:] = [0.1,0.1,0.1,0.0,0.0,0.2,0.2,0.1,0.2,0.1,0.95,0.1,0.1,0.2,0.1,0.0,0.3,0.1,0.4]
obj_room_sc_ne[9,:] = [0.99,0.98,0.97,0.99,0.99,0.95,0.9,0.98,0.9,0.95,0.2,0.98,0.98,0.95,0.97,0.99,0.85,0.97,0.7]
# obj_room_sc[10,:] = [0.1, 0.7, 0.6, 0.2, 0.1, 0.2, 0.0, 0.1, 0.0, 0.9, 0.1, 0.8, 0.3, 0.0, 0.5, 0.1, 0.4, 0.7, 0.0]
obj_room_sc_po[10,:] = [0.1,0.7,0.6,0.2,0.3,0.4,0.5,0.3,0.2,0.8,0.3,0.7,0.4,0.2,0.6,0.1,0.5,0.6,0.2]
obj_room_sc_ne[10,:] = [0.95,0.5,0.6,0.7,0.9,0.8,0.7,0.85,0.9,0.3,0.8,0.6,0.85,0.9,0.7,0.95,0.7,0.6,0.85]
# obj_room_sc[11,:] =  [0.2, 0.9, 0.8, 0.1, 0.3, 0.7, 0.0, 0.4, 0.1, 0.9, 0.8, 0.9, 0.7, 0.1, 0.6, 0.2, 0.1, 0.8, 0.1]
obj_room_sc_po[11,:] = [0.5,0.7,0.6,0.2,0.4,0.6,0.3,0.3,0.3,0.6,0.5,0.7,0.5,0.2,0.4,0.3,0.2,0.6,0.4]
obj_room_sc_ne[11,:] = [0.7,0.5,0.6,0.9,0.8,0.6,0.8,0.7,0.85,0.4,0.6,0.4,0.7,0.9,0.6,0.9,0.8,0.5,0.75]
# obj_room_sc[12,:] = [0.1, 0.7, 0.5, 0.1, 0.2, 0.9, 0.0, 0.2, 0.1, 0.6, 0.4, 0.9, 0.8, 0.1, 0.5, 0.2, 0.1, 0.7, 0.1]
obj_room_sc_po[12,:] = [0.2,0.5,0.4,0.1,0.3,0.6,0.2,0.2,0.1,0.4,0.4,0.6,0.4,0.1,0.3,0.2,0.1,0.5,0.2]
obj_room_sc_ne[12,:] = [0.7,0.5,0.6,0.8,0.8,0.5,0.8,0.85,0.9,0.6,0.6,0.4,0.75,0.85,0.7,0.9,0.8,0.6,0.85]
# obj_room_sc[13,:] = [0.2, 0.4, 0.3, 0.2, 0.1, 0.9, 0.0, 0.2, 0.1, 0.7, 1.0, 0.8, 0.3, 0.6, 0.4, 0.1, 0.2, 0.5, 0.2]
obj_room_sc_po[13,:] = [0.3,0.5,0.4,0.1,0.2,0.6,0.2,0.2,0.2,0.4,0.8,0.6,0.3,0.2,0.3,0.1,0.2,0.5,0.3]
obj_room_sc_ne[13,:] = [0.6,0.5,0.5,0.8,0.8,0.4,0.7,0.7,0.85,0.5,0.3,0.4,0.7,0.8,0.6,0.85,0.75,0.5,0.65]
# obj_room_sc[14,:] = [0.8, 0.3, 0.2, 0.4, 0.1, 0.7, 0.0, 0.3, 0.5, 0.4, 1.0, 0.6, 0.2, 0.9, 0.3, 0.1, 0.5, 0.4, 0.6]
obj_room_sc_po[14,:] = [0.4,0.6,0.5,0.2,0.3,0.5,0.3,0.4,0.3,0.4,0.7,0.5,0.3,0.2,0.4,0.2,0.3,0.5,0.3]
obj_room_sc_ne[14,:] = [0.6,0.5,0.5,0.7,0.75,0.6,0.7,0.6,0.7,0.5,0.4,0.5,0.65,0.75,0.6,0.8,0.65,0.6,0.5]



obj_room_sc = obj_room_sc_po - obj_room_sc_ne

def is_int(s):
    try:
        int(s)
        return True
    except:
        return False


class SemanticMapDataset(Dataset):
    grid_size = 0.05 # m
    object_boundary = 1.0 # m
    def __init__(
        self,
        cfg,
        split='train',
        scf_name=None,
        seed=None,
    ):
        self.cfg = cfg
        self.dset = cfg.dset_name
        # Seed the dataset
        if seed is None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
        else:
            random.seed(seed)
            np.random.seed(seed)
        # Load maps
        maps_path = sorted(glob.glob(osp.join(cfg.root, "*.h5")))
        # print("map_path in dataset.py=",maps_path)
        # maps_path=""
        maps_path=["data/semantic_maps/gibson/semantic_maps/Darden.h5"]
        
        maps_path_room=["data/semantic_maps/gibson/semantic_maps/Darden_room.h5"]

        # Load json info
        maps_info = json.load(open(osp.join(cfg.root, 'semmap_GT_info.json')))
        # print("maps_info=",maps_info)

        print("OPEN OBJ H5 FILE @@@@@@@@@@@")
        maps = {}
        names = []
        maps_xyz_info = {}
        for path in maps_path:
            scene_name = path.split('/')[-1].split('.')[0]
            print("scene_name1=",scene_name)
            if scene_name not in SPLIT_SCENES[self.dset][split]:
                continue
            if (scf_name is not None) and (scene_name not in scf_name):
                continue
            with h5py.File(path, 'r') as fp:
                floor_ids = sorted([key for key in fp.keys() if is_int(key)])
                for floor_id in floor_ids:
                    name = f'{scene_name}_{floor_id}'
                    if (scf_name is not None) and (name != scf_name):
                        continue
                    # print("scene_name2=",scene_name)
                    map_world_shift = maps_info[scene_name]['map_world_shift']
                    # print("map_world_shift=",map_world_shift)
                    if floor_id not in maps_info[scene_name]:
                        continue
                    map_y = maps_info[scene_name][floor_id]['y_min']
                    resolution = maps_info[scene_name]['resolution']
                    map_semantic = np.array(fp[floor_id]['map_semantic'])
                    # print("map_semantic=",map_semantic.shape)
                    nuniq = len(np.unique(map_semantic))
                    if nuniq >= MIN_OBJECTS_THRESH + 2:
                        # print("nuniq=",nuniq)
                        # print("name=",name)
                        names.append(name)
                        maps[name] = self.convert_maps_to_oh(map_semantic)
                        # print("maps[name]=",maps[name].shape)
                        maps_xyz_info[name] = {
                            'world_shift': map_world_shift,
                            'resolution': resolution,
                            'y': map_y,
                            'scene_name': scene_name,
                        }
            # open room h5 file
        print("OPEN ROOM H5 FILE ###########")
        maps_room = {}
        names_room = []
        maps_xyz_info_room = {}
        for path in maps_path_room:
            scene_name = path.split('/')[-1].split('.')[0]
            scene_name = scene_name.replace("_room", "")
            # print("scene_name1_room=",scene_name)
            if scene_name not in SPLIT_SCENES[self.dset][split]:
                # print("continue0")
                continue
            if (scf_name is not None) and (scene_name not in scf_name):
                continue
            with h5py.File(path, 'r') as fp:
                # print("path_room=",path)
                floor_ids = sorted([key for key in fp.keys() if is_int(key)])
                # print("floor_ids_room=",floor_ids)
                for floor_id in floor_ids:
                    # print("floor_id_room=",floor_id)
                    name = f'{scene_name}_{floor_id}'
                    # print("name_room=",name)
                    if (scf_name is not None) and (name != scf_name):
                        # print("continue1!!!!!")
                        continue
                    # print("scene_name_room=",scene_name)
                    map_world_shift = maps_info[scene_name]['map_world_shift']
                    # print("map_world_shift_room=",map_world_shift)
                    if floor_id not in maps_info[scene_name]:
                        # print("continue2!!!!!")
                        continue
                    map_y = maps_info[scene_name][floor_id]['y_min']
                    resolution = maps_info[scene_name]['resolution']
                    map_semantic = np.array(fp[floor_id]['map_semantic'])
                    # print("map_semantic_room=",map_semantic.shape)
                    nuniq = len(np.unique(map_semantic))
                    # print("nuniq_room=",nuniq)
                    if nuniq >= MIN_OBJECTS_THRESH + 2:
                        # print("name_room2=",name)
                        # print("name_room=",name)
                        names_room.append(name)
                        # print("map_semantic_room=",map_semantic.shape)

                        maps_room[name] = self.convert_maps_to_oh_room(map_semantic)
                        # print("maps[name]_room=",maps_room[name].shape)
                        maps_xyz_info_room[name] = {
                            'world_shift': map_world_shift,
                            'resolution': resolution,
                            'y': map_y,
                            'scene_name': scene_name,
                        }

        
        self.maps = maps
        self.maps_room = maps_room
        # print("self.maps=",maps.keys())
        # print("self.maps_room=",maps_room.keys())
        # print("self map shape=",self.maps_room, self.maps_room.shape,)
        self.names = sorted(names)
        # print(" self.names =", self.names )
        self.maps_xyz_info = maps_xyz_info
        self.visibility_size = cfg.visibility_size
        # Pre-compute FMM dists for each semmap
        if self.cfg.fmm_dists_saved_root == '':
            self.fmm_dists = self.compute_fmm_dists()
        else:
            self.fmm_dists = {}
            for name in self.names:
                fname = f'{cfg.fmm_dists_saved_root}/{name}.pbz2'
                with bz2.BZ2File(fname, 'rb') as fp:
                    self.fmm_dists[name] = (cPickle.load(fp)).astype(np.float32)
        # Pre-compute navigable locations for each map
        self.nav_locs = self.compute_navigable_locations()
        # array1, array2 = self.nav_locs
        # print("self.nav_locs=", self.nav_locs)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        name = self.names[idx]
        # print("idx=",idx)
        # print("name in getitem=",name)
        semmap = self.maps[name]
        semmap_room = self.maps_room[name]
        # print("semmap in getitem=",semmap.shape)
        # print("semmap room in getitem=",self.maps_room[name].shape)
        fmm_dists = self.fmm_dists[name]
        map_xyz_info = self.maps_xyz_info[name]
        # print("map_xyz_info=",map_xyz_info)
        nav_space = semmap[FLOOR_ID]
        nav_locs = self.nav_locs[name]
        # print("nav_locs=",nav_locs,nav_locs[0].shape)
        # Create input and output maps
        if self.cfg.masking_mode == 'spath':
            spath = self.get_random_shortest_path(nav_space, nav_locs)
            # print("spath=",spath)
            input, label = self.create_spath_based_input_output_pairs(
                semmap, fmm_dists, spath, map_xyz_info,
            )
            # print("input_obj=",input.shape,label.keys())
            input_room, label_room = self.create_spath_based_input_output_pairs_room(
                semmap_room, fmm_dists, spath, map_xyz_info,
            )
            # print("input_room=",input_room.shape,label_room.keys())
            
        else:
            raise ValueError(f"Masking mode {self.cfg.masking_mode} is not implemented!")
        return input, input_room, label, label_room

    def get_item_by_name(self, name):
        assert name in self.names
        idx = self.names.index(name)
        return self[idx]

    def convert_maps_to_oh(self, semmap):
        ncat = NUM_OBJECT_CATEGORIES[self.dset]
        # print("ncat=",ncat)
        semmap_oh = np.zeros((ncat, *semmap.shape), dtype=np.float32)
        for i in range(0, ncat):
            # print(i)
            semmap_oh[i] = (semmap == i + CAT_OFFSET).astype(np.float32)
        return semmap_oh
    

    def convert_maps_to_oh_room(self, semmap):
        ncat = 21
        semmap_oh = np.zeros((ncat, *semmap.shape), dtype=np.float32)
        for i in range(0, ncat):
            # print(i)
            semmap_oh[i] = (semmap == i + CAT_OFFSET).astype(np.float32)
        return semmap_oh





    def plan_path(self, nav_space, start_loc, end_loc):
        planner = FMMPlanner(nav_space)
        planner.set_goal(end_loc)
        curr_loc = start_loc
        spath = [curr_loc]
        ctr = 0
        while True:
            ctr += 1
            if ctr > 10000:
                print("plan_path() --- Run into infinite loop!")
                break
            next_y, next_x, _, stop = planner.get_short_term_goal(curr_loc)
            if stop:
                break
            curr_loc = (next_y, next_x)
            spath.append(curr_loc)
        return spath

    def get_random_shortest_path(self, nav_space, nav_locs):
        planner = FMMPlanner(nav_space)
        ys, xs = nav_locs
        num_outer_trials = 0
        while True:
            num_outer_trials += 1
            if num_outer_trials > 1000:
                print(f"=======> Stuck in infinite outer loop in!")
                break
            # Pick a random start location
            rnd_ix = np.random.randint(0, xs.shape[0])
            start_x, start_y = xs[rnd_ix], ys[rnd_ix]
            planner.set_goal((start_y, start_x))
            # Ensure that this is reachable from other points in the scene
            rchble_mask = planner.fmm_dist < planner.fmm_dist.max().item()
            if np.count_nonzero(rchble_mask) < 20:
                continue
            rchble_y, rchble_x = np.where(rchble_mask)
            # Pick a random goal location
            rnd_ix = np.random.randint(0, rchble_x.shape[0])
            end_x, end_y = rchble_x[rnd_ix], rchble_y[rnd_ix]
            # print("end_y=",end_y)

            break
        # Plan path from start to goal
        # print("end_y=",end_y)
        spath = self.plan_path(nav_space, (start_y, start_x), (end_y, end_x))
        return spath

    def compute_fmm_dists(self):
        fmm_dists = {}
        selem = skmp.disk(int(self.object_boundary / self.grid_size))
        # print("selem=",selem.shape)
        # print("self.names=",self.names)
        for name in tqdm.tqdm(self.names):
            # print("name in compute fmm=",name)
            semmap = self.maps[name]
            # print("FLOOR_ID=",FLOOR_ID)
            navmap = semmap[FLOOR_ID]
            dists = []
            for catmap in semmap:
                if np.count_nonzero(catmap) == 0:
                    fmm_dist = np.zeros(catmap.shape)
                    fmm_dist.fill(np.inf)
                else:
                    cat_navmap = skmp.binary_dilation(catmap, selem) != True
                    cat_navmap = 1 - cat_navmap
                    cat_navmap[navmap > 0] = 1
                    planner = FMMPlanner(cat_navmap)
                    planner.set_multi_goal(catmap)
                    fmm_dist = np.copy(planner.fmm_dist)
                dists.append(fmm_dist)
            fmm_dists[name] = np.stack(dists, axis=0).astype(np.float32)
        return fmm_dists

    def compute_object_pfs(self, fmm_dists):
        cutoff = self.cfg.object_pf_cutoff_dist
        opfs = torch.clamp((cutoff - fmm_dists) / cutoff, 0.0, 1.0)
        return opfs

    def compute_navigable_locations(self):
        nav_locs = {}
        for name in self.names:
            semmap = self.maps[name]
            navmap = semmap[FLOOR_ID]
            ys, xs = np.where(navmap)
            nav_locs[name] = (ys, xs)
        return nav_locs

    def get_world_coordinates(self, map_xy, world_xyz_info):
        shift_xyz = world_xyz_info['world_shift']
        resolution = world_xyz_info['resolution']
        world_y = world_xyz_info['y']
        world_xyz = (
            map_xy[0] * resolution + shift_xyz[0],
            world_y,
            # ，y 坐标可能表示一个固定的高度或层面。例如，在2.5D表示或者某些特定的3D环境中，y 坐标可能用于表示固定的地面或者某个
            map_xy[1] * resolution + shift_xyz[2],
        )
        return world_xyz

    def get_visibility_map(self, in_semmap, locations):
        """
        locations - list of [y, x] coordinates
        """
        vis_map = np.zeros(in_semmap.shape[1:], dtype=np.uint8)
        for i in range(len(locations)):
            y, x = locations[i]
            y, x = int(y), int(x)
            if self.cfg.masking_shape == 'square':
                S = int(self.visibility_size / self.grid_size / 2.0)
                vis_map[(y - S) : (y + S), (x - S) : (x + S)] = 1
            else:
                raise ValueError(f'Masking shape {self.cfg.masking_shape} not defined!')

        vis_map = torch.from_numpy(vis_map).float()
        return vis_map

    def create_spath_based_input_output_pairs(
        self, semmap, fmm_dists, spath, map_xyz_info
    ):
        out_semmap = torch.from_numpy(semmap)
        out_fmm_dists = torch.from_numpy(fmm_dists) * self.grid_size
        in_semmap = out_semmap.clone()
        vis_map = self.get_visibility_map(in_semmap, spath)
        in_semmap *= vis_map


        global tform_trans, tform_rot

        # Transform the maps about a random center and rotate by a random angle
        center = random.choice(spath)
        rot = random.uniform(-math.pi, math.pi)
        Wby2, Hby2 = out_semmap.shape[2] // 2, out_semmap.shape[1] // 2
        tform_trans = torch.Tensor([[center[1] - Wby2, center[0] - Hby2, 0]])
        tform_rot = torch.Tensor([[0, 0, rot]])

        # print ("in_semmap1 =",in_semmap.shape,out_semmap.shape,out_fmm_dists.shape)
        (
            in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks
        ) = self.transform_input_output_pairs(
            in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot)
        # print("in_semmap2=",in_semmap.shape,out_semmap.shape,out_fmm_dists.shape)



        # Get real-world position and orientation of agent
        world_xyz = self.get_world_coordinates(center, map_xyz_info)
        world_heading = -rot # Agent turning leftward is positive in habitat
        scene_name = map_xyz_info['scene_name']
        print("scene_name=",scene_name)
        # map_xyz_info= {'world_shift': [-21.07318115234375, 0.0, -8.210206031799316], 'resolution': 0.05, 'y': 2.8223652625215943, 'scene_name': 'Darden'}
        print("map_xyz_info##################=",map_xyz_info)
        object_pfs = self.compute_object_pfs(out_fmm_dists)
        return in_semmap, {
            'semmap': out_semmap,
            'fmm_dists': out_fmm_dists,
            'agent_fmm_dist': agent_fmm_dist,
            'object_pfs': object_pfs,
            'masks': out_masks,
            'world_xyz': world_xyz,
            'world_heading': world_heading,
            'scene_name': scene_name,
        }
    

    # return input and label for room map
    def create_spath_based_input_output_pairs_room(
        self, semmap_room, fmm_dists, spath, map_xyz_info
    ):
        out_semmap = torch.from_numpy(semmap_room)
        out_fmm_dists = torch.from_numpy(fmm_dists) * self.grid_size
        in_semmap = out_semmap.clone()
        vis_map = self.get_visibility_map(in_semmap, spath)
        in_semmap *= vis_map
        # Transform the maps about a random center and rotate by a random angle
        center = random.choice(spath)
        rot = random.uniform(-math.pi, math.pi)
        Wby2, Hby2 = out_semmap.shape[2] // 2, out_semmap.shape[1] // 2
        # tform_trans = torch.Tensor([[center[1] - Wby2, center[0] - Hby2, 0]])
        # tform_rot = torch.Tensor([[0, 0, rot]])
        (
            in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks
        ) = self.transform_input_output_pairs(
            in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot)




        # Get real-world position and orientation of agent
        world_xyz = self.get_world_coordinates(center, map_xyz_info)
        world_heading = -rot # Agent turning leftward is positive in habitat
        scene_name = map_xyz_info['scene_name']
        object_pfs = self.compute_object_pfs(out_fmm_dists)


        return in_semmap, {
            'semmap': out_semmap,
            'fmm_dists': out_fmm_dists,
            'agent_fmm_dist': agent_fmm_dist,
            'object_pfs': object_pfs,
            'masks': out_masks,
            'world_xyz': world_xyz,
            'world_heading': world_heading,
            'scene_name': scene_name,
        }





    def transform_input_output_pairs(
        self, in_semmap, out_semmap, out_fmm_dists, tform_trans, tform_rot
    ):
        # Invert fmm_dists for transformations (since padding is zeros)
        max_dist = out_fmm_dists[out_fmm_dists != math.inf].max() + 1
        out_fmm_dists = 1 / (out_fmm_dists + EPS)
        # Expand to add batch dim
        in_semmap = in_semmap.unsqueeze(0)
        out_semmap = out_semmap.unsqueeze(0)
        out_fmm_dists = out_fmm_dists.unsqueeze(0)
        # Crop a large-enough map around agent
        _, N, H, W = in_semmap.shape
        crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
        map_size = int(2.0 * self.cfg.output_map_size / self.grid_size)
        in_semmap = crop_map(in_semmap, crop_center, map_size)
        out_semmap = crop_map(out_semmap, crop_center, map_size)
        out_fmm_dists = crop_map(out_fmm_dists, crop_center, map_size)
        # Rotate the map
        in_semmap = spatial_transform_map(in_semmap, tform_rot)
        out_semmap = spatial_transform_map(out_semmap, tform_rot)
        out_fmm_dists = spatial_transform_map(out_fmm_dists, tform_rot)
        # Crop out the appropriate size of the map
        _, N, H, W = in_semmap.shape
        map_center = torch.Tensor([[W / 2.0, H / 2.0]])
        map_size = int(self.cfg.output_map_size / self.grid_size)
        in_semmap = crop_map(in_semmap, map_center, map_size, 'nearest')
        out_semmap = crop_map(out_semmap, map_center, map_size, 'nearest')
        out_fmm_dists = crop_map(out_fmm_dists, map_center, map_size, 'nearest')
        # Create a scaling-mask for the loss function. By default, select
        # only navigable / object regions (where fmm_dists exists).
        out_masks = (out_semmap[0, FLOOR_ID] >= 0.5).float() # (H, W)
        out_masks = repeat(out_masks, 'h w -> () n h w', n=N)
        # Mask out potential fields based on input regions
        if self.cfg.potential_function_masking:
            # Compute frontier locations
            unk_map = (
                torch.max(in_semmap, dim=1).values[0] < 0.5
            ).float().numpy() # (H, W)
            free_map = (in_semmap[0, FLOOR_ID] >= 0.5).float().numpy() # (H, W)
            frontiers = get_frontiers_np(unk_map, free_map)
            frontiers = torch.from_numpy(frontiers).float().unsqueeze(0).unsqueeze(1)

            # Dilate the frontiers mask
            frontiers_mask = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3)
            # Scaling loss at the frontiers
            alpha = self.cfg.potential_function_frontier_scaling
            # Scaling loss at the non-visible regions
            beta = self.cfg.potential_function_non_visible_scaling
            visibility_mask = (in_semmap.sum(dim=1, keepdim=True) > 0).float()
            # Scaling loss at the visible & non-frontier regions
            gamma = self.cfg.potential_function_non_frontier_scaling
            not_frontier_or_visible = (1 - visibility_mask) * (1 - frontiers_mask)
            visible_and_not_frontier = visibility_mask * (1 - frontiers_mask)
            # Compute final mask
            out_masks = out_masks *(
                visible_and_not_frontier * gamma + \
                not_frontier_or_visible * beta + \
                frontiers_mask * alpha
            )
        # Remove batch dim
        in_semmap = in_semmap.squeeze(0)
        out_semmap = out_semmap.squeeze(0)
        out_fmm_dists = out_fmm_dists.squeeze(0)
        out_masks = out_masks.squeeze(0)
        # Invert fmm_dists for transformations (since padding, new pixels, etc are zeros)
        out_fmm_dists = torch.clamp(1 / (out_fmm_dists + EPS), 0.0, max_dist)
        # Compute distance from agent to all locations on the map
        nav_map = out_semmap[FLOOR_ID].numpy() # (H, W)
        planner = FMMPlanner(nav_map)
        agent_map = np.zeros(nav_map.shape, dtype=np.float32)
        Hby2, Wby2 = agent_map.shape[0] // 2, agent_map.shape[1] // 2
        agent_map[Hby2 - 1 : Hby2 + 2, Wby2 - 1 : Wby2 + 2] = 1
        selem = skmp.disk(int(self.object_boundary / 2.0 / self.grid_size))
        agent_map = skmp.binary_dilation(agent_map, selem) != True
        agent_map = 1 - agent_map
        planner.set_multi_goal(agent_map)
        agent_fmm_dist = torch.from_numpy(planner.fmm_dist) * self.grid_size

        return in_semmap, out_semmap, out_fmm_dists, agent_fmm_dist, out_masks

    @staticmethod
    def visualize_map(semmap, bg=1.0, dataset='gibson'):
        n_cat = semmap.shape[0] - 2 # Exclude floor and wall
        def compress_semmap(semmap):
            c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
            for i in range(semmap.shape[0]):
                c_map[semmap[i] > 0.] = i+1
            return c_map

        palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]
        if dataset == 'gibson':
            palette += [int(x * 255.) for x in gibson_palette[15:]]
        else:
            palette += [c for color in d3_40_colors_rgb[:n_cat]
                        for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        c_map_arry=np.array(c_map)
        # print(c_map_arry.shape)
        # print("OBJ=",np.unique(c_map_arry))
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)

        return semantic_img


    @staticmethod
    def visualize_map_room(semmap, bg=1.0, dataset='gibson'):
        n_cat = semmap.shape[0] - 2 # Exclude floor and wall
        # print("n_cat=",n_cat) #19 =21-2

        # global obj_room_sc
        # print(obj_room_sc.shape,obj_room_sc)

        # print(GIBSON_ROOM_CATEGORIES[0])
        # print(GIBSON_OBJECT_CATEGORIES[0])

        def compress_semmap(semmap):
            c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
            for i in range(semmap.shape[0]):
                # print("semmap[i]=",semmap[i].shape)
                c_map[semmap[i] > 0.] = i+1
            # print("c_map=",c_map.shape)
            c_map_arry_obj=np.array(c_map)
            # print(np.unique(c_map_arry_obj))
            return c_map

        palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]
        if dataset == 'gibson':
            # print("ROOM COLOR DATASET=",dataset)
            palette += [int(x * 255.) for x in gibson_palette_room[15:]]




        # else:
        #     palette += [c for color in d3_40_colors_rgb[:n_cat]
        #                 for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        # plt.imshow(c_map)
        # plt.savefig("c_map_room.png")
        # print(np.array(c_map))
        c_map_arry=np.array(c_map)
        # print("ROOM=",np.unique(c_map_arry))
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)
        # print("semantic_img=",semantic_img.shape)
        return semantic_img
    

    @staticmethod
    def visualize_map_obj_room_sc(semmap, semmap_room, bg=1.0, dataset='gibson'):
        n_cat = semmap.shape[0] - 2 # Exclude floor and wall
        # print("n_cat=",n_cat) #19 =21-2

        global obj_room_sc
        # print(obj_room_sc.shape,obj_room_sc[0])
# 
        # print(GIBSON_ROOM_CATEGORIES[0])
        # print(GIBSON_OBJECT_CATEGORIES[0])

        # print("semmap=",semmap.shape)
        # print("semmap_room=",semmap_room.shape)

        tgt_map = semmap[2]
        # print("tgt_map=",np.array(tgt_map))      
        # plt.imshow(tgt_map)
        # plt.savefig("tgt_map.png")  

        tgt_room_map=semmap_room[2]
        # print("tgt_room_map=",np.array(tgt_room_map))
        # plt.imshow(tgt_room_map)
        # plt.savefig("tgt_room_map.png")  

        # likelihood_values = [0.1, 0.9, 0.8, 0.1, 0.2, 1.0, 0.0, 0.5, 0.2, 0.9, 0.7, 1.0, 0.8, 0.2, 0.7, 0.1, 0.2, 1.0, 0.2]
        obj_room_sc_map = np.zeros((480, 480))


        #visualize room obj matrix 
        # plt.imshow(obj_room_sc, cmap='viridis', aspect='auto')
        # plt.colorbar(label='Value')
        # plt.title('Visualization of obj_room_sc Matrix')
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')

        # plt.show()



        for i in range(2, 21):
            mask = semmap_room[i] == 1
            obj_room_sc_map = np.add(obj_room_sc_map, mask * obj_room_sc[14,i-2])

        plt.imshow(obj_room_sc_map, cmap='YlGnBu',  vmin=-1, vmax=1, origin='upper')
        plt.colorbar()
        # plt.title('Object-Room Likelihood Map')
        plt.savefig('/home/aae14859ln/Sun/PONI/ZZZZZZZobj_room_sc_map_chairs_new.png', dpi=300)
        plt.show()



        def compress_semmap(semmap):
            c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
            for i in range(semmap.shape[0]):
                # print("semmap[i]=",semmap[i].shape)
                c_map[semmap[i] > 0.] = i+1
            # print("c_map=",c_map.shape)
            c_map_arry_obj=np.array(c_map)
            # print(np.unique(c_map_arry_obj))
            return c_map

        palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]
        if dataset == 'gibson':
            print("ROOM COLOR DATASET=",dataset)
            palette += [int(x * 255.) for x in gibson_palette_room[15:]]




        # else:
        #     palette += [c for color in d3_40_colors_rgb[:n_cat]
        #                 for c in color.tolist()]
        semmap = asnumpy(semmap)
        c_map = compress_semmap(semmap)
        plt.imshow(c_map)
        plt.savefig("c_map_room.png")
        # print(np.array(c_map))
        c_map_arry=np.array(c_map)
        # print("ROOM=",np.unique(c_map_arry))
        semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
        semantic_img.putpalette(palette)
        semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img)
        # print("semantic_img=",semantic_img.shape)
        return semantic_img





    @staticmethod
    def visualize_object_pfs(
        in_semmap, semmap, object_pfs, dirs=None, locs=None, dataset='gibson'
    ):
        """
        semmap - (C, H, W)
        object_pfs - (C, H, W)
        """
        in_semmap = asnumpy(in_semmap)
        semmap = asnumpy(semmap)
        semmap_rgb = SemanticMapDataset.visualize_map(in_semmap, bg=1.0, dataset=dataset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        object_pfs = asnumpy(object_pfs)
        vis_ims = []
        for i in range(0, semmap.shape[0]):
            opf = object_pfs[i][..., np.newaxis]
            sm = np.copy(semmap_rgb)
            smpf = red_image * opf + sm * (1 - opf)
            smpf = smpf.astype(np.uint8)
            # Highlight directions
            if dirs is not None and dirs[i] is not None:
                dir = math.radians(dirs[i])
                sx, sy = sm.shape[1] // 2, sm.shape[0] // 2
                ex = int(sx + 200 * math.cos(dir))
                ey = int(sy + 200 * math.sin(dir))
                cv2.line(smpf, (sx, sy), (ex, ey), (0, 255, 0), 3)
            # Highlight object locations
            smpf[semmap[i] > 0, :] = np.array([0, 0, 255])
            # Highlight location targets
            if locs is not None:
                H, W = semmap.shape[1:]
                x, y = locs[i]
                if x >= 0 and y >= 0:
                    x, y = int(x * W), (y * H)
                    cv2.circle(smpf, (x, y), 3, (0, 255, 0), -1)
            vis_ims.append(smpf)

        return vis_ims





    # visualize object-room pf

    @staticmethod
    def visualize_object_pfs_room(
        in_semmap,in_semmap_room, semmap, semmap_room, object_pfs, dirs=None, locs=None, dataset='gibson'
    ):
        """
        semmap - (C, H, W)
        object_pfs - (C, H, W)
        """
        # print("######################")
        global obj_room_sc
        # print("obj_room_sc in visualization",obj_room_sc.shape,obj_room_sc[0])

        in_semmap = asnumpy(in_semmap)
        in_semmap_room = asnumpy(in_semmap_room)
        semmap = asnumpy(semmap)
        semmap_room = asnumpy(semmap_room)
        semmap_rgb = SemanticMapDataset.visualize_map(in_semmap, bg=1.0, dataset=dataset)
        semmap_rgb_room = SemanticMapDataset.visualize_map_room(in_semmap_room, bg=1.0, dataset=dataset)

        red_image = np.zeros_like(semmap_rgb)
        red_image_room = np.zeros_like(semmap_rgb_room)

        red_image[..., 0] = 255
        # red_image_room[..., ...] = 255
        black_image = np.zeros_like(red_image)

        green_image = np.zeros_like(red_image)  # 创建一个与red_image形状相同的新图像
        green_image[...,1] = 255  # 设置绿色通道为最大值

        object_pfs = asnumpy(object_pfs)
        print("object_pfs=",object_pfs.shape)
        plt.imshow(object_pfs[0])
        plt.savefig("object_pfs_[0].png")

        obj_room_sc_map = np.zeros((480, 480))
        obj_room_sc_maps = np.zeros((15,480,480))
        result_obj_room_maps = np.zeros((17,480,480))


        for m in range (0,15):
            obj_room_sc_map.fill(0)
            # print("m=",m) m=0-14
            for i in range(2, 21):
                mask = semmap_room[i] == 1
                obj_room_sc_map = np.add(obj_room_sc_map, mask * obj_room_sc[m,i-2])
                # print("obj_room_sc_map=",obj_room_sc_map.shape)
            # plt.imshow(obj_room_sc_map,cmap='YlGnBu', origin='upper')
            # plt.colorbar(label='Room-object relation Value')
            # plt.savefig('obj_room_sc_map_1.png', dpi=300)

                frontier_mask = object_pfs[0] > 0
                obj_room_sc_frontier_map = np.where(frontier_mask,obj_room_sc_map,0)
                # print("obj_room_sc_frontier+map=",obj_room_sc_frontier_map.shape)
        
                # plt.imshow(obj_room_sc_frontier_map,cmap='YlGnBu', origin='upper')
                # plt.colorbar(label='Room-object relation Value')
                # plt.savefig('obj_room_sc_frontier_map_14.png', dpi=300)
            obj_room_sc_maps[m] = obj_room_sc_frontier_map
        # print("obj_room_sc_maps shape ==",obj_room_sc_maps.shape)

        result_obj_room_maps[:2] = object_pfs[:2]
        result_obj_room_maps[2:] = obj_room_sc_maps

            
        # plt.imshow(obj_room_sc_maps[1],cmap='YlGnBu', origin='upper')
        # plt.colorbar(label='Room-object relation Value')
        # plt.title('Partial object-room frontier map (couch)')
        # plt.savefig('zzzzzz2obj_room_sc_maps_bottle.png', dpi=300)

    
        # plt.imshow(result_obj_room_maps[16],cmap='YlGnBu', origin='upper')
        # plt.colorbar(label='Room-object relation Value')
        # plt.savefig('result_obj_room_maps[16].png', dpi=300)

        # print("result_obj_room_maps=",result_obj_room_maps.shape)
        

      

        vis_ims = []
        for i in range(0, semmap.shape[0]):
            # plt.imshow(obj_room_sc_maps[4],cmap='YlGnBu', origin='upper')
            # # plt.colorbar(label='Room-object relation Value')
            # plt.savefig('obj_room_sc_maps[4].png', dpi=300)
            opf = result_obj_room_maps[i][..., np.newaxis]
            sm = np.copy(semmap_rgb_room)

            # plt.imshow(opf,cmap='YlGnBu',vmin=-1,vmax=1, origin='upper')
            # plt.colorbar()
            # plt.savefig(str(i)+'ZZZZZZZZZopf.png', dpi=300)

            # print("red_image=",red_image.shape)
            # print("opf=",opf.shape)
            # # smpf = red_image * opf 
            # smpf = red_image * opf + sm * (1 - opf)
            # smpf = smpf.astype(np.uint8)
            # # Highlight object locations
            # smpf[semmap[i] > 0, :] = np.array([0, 0, 255])

            # 将opf从 (480, 480, 1) 扩展到 (480, 480, 3)
            opf_expanded = np.repeat(opf, 3, axis=2)

            sm = np.copy(semmap_rgb_room)

            mask_negative = opf_expanded < 0  # [-1, 0)范围的掩码
            mask_positive = opf_expanded >= 0  # [0, 1]范围的掩码

            # 应用掩码到图片上
            smpf = np.zeros_like(red_image)
            smpf[mask_negative] = black_image[mask_negative] * opf_expanded[mask_negative]
            smpf[mask_positive] = red_image[mask_positive] * opf_expanded[mask_positive]

            # 计算最终结果
            smpf = smpf + sm * (1 - opf_expanded)
            smpf = smpf.astype(np.uint8)

            # 高亮显示对象位置
            smpf[semmap[i] > 0, :] = np.array([0, 0, 255])

  
    
            vis_ims.append(smpf)


        # 这个是partial room segmentation in rgb  没有显示frontier的颜色 = semmap_rgb_room
        # plt.imshow(semmap_rgb_room,cmap='YlGnBu',vmin=-1,vmax=1, origin='upper')
        # plt.colorbar()
        # plt.savefig('ZZZZZZZZZsemmap_rgb_room.png', dpi=300)

        return vis_ims









    @staticmethod
    def visualize_object_category_pf(semmap, object_pfs, cat_id, dset):
        """
        semmap - (C, H, W)
        object_pfs - (C, H, W)
        cat_id - integer
        """
        semmap = asnumpy(semmap)
        offset = OBJECT_CATEGORIES[dset].index('chair')
        object_pfs = asnumpy(object_pfs)[cat_id + offset] # (H, W)
        object_pfs = object_pfs[..., np.newaxis] # (H, W)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap, bg=1.0, dataset=dset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        smpf = red_image * object_pfs + semmap_rgb * (1 - object_pfs)
        smpf = smpf.astype(np.uint8)

        return smpf

    def visualize_area_pf(semmap, area_pfs, dset='gibson'):
        """
        semmap - (C, H, W)
        are_pfs - (1, H, W)
        """
        semmap = asnumpy(semmap)
        pfs = asnumpy(area_pfs)[0] # (H, W)
        pfs = pfs[..., np.newaxis] # (H, W)
        semmap_rgb = SemanticMapDataset.visualize_map(semmap, bg=1.0, dataset=dset)
        red_image = np.zeros_like(semmap_rgb)
        red_image[..., 0] = 255
        smpf = red_image * pfs + semmap_rgb * (1 - pfs)
        smpf = smpf.astype(np.uint8)
        
        return smpf

    @staticmethod
    def combine_image_grid(
        in_semmap, out_semmap, gt_object_pfs, pred_object_pfs=None,
        gt_acts=None, gt_area_pfs=None, pred_area_pfs=None, dset=None,
        n_per_row=8, pad=2, border_color=200, output_width=1024,
    ):
        img_and_titles = [
            (in_semmap, 'Input map'), (out_semmap, 'Full output map')
        ]
        if gt_area_pfs is not None:
            img_and_titles.append((gt_area_pfs, 'GT Area map'))
        if pred_area_pfs is not None:
            img_and_titles.append((pred_area_pfs, 'Pred Area map'))
        for i, cat in INV_OBJECT_CATEGORY_MAP[dset].items():
            acts_suffix = ''
            if gt_acts is not None:
                acts_suffix = f'(act: {gt_acts[i].item():d})'
            if cat in ['wall', 'floor']:
                continue
            if pred_object_pfs is None:
                title = 'PF for ' + cat + acts_suffix
                img_and_titles.append((gt_object_pfs[i], title))
            else:
                title = 'GT PF for ' + cat + acts_suffix
                img_and_titles.append((gt_object_pfs[i], title))
                title = 'Pred PF for ' + cat + acts_suffix
                img_and_titles.append((pred_object_pfs[i], title))

        imgs = []
        for img, title in img_and_titles:
            cimg = SemanticMapDataset.add_title_to_image(img, title)
            # Pad image
            cimg = np.pad(cimg, ((pad, pad), (pad, pad), (0, 0)),
                          mode='constant', constant_values=border_color)
            imgs.append(cimg)

        # Convert to grid
        n_rows = len(imgs) // n_per_row
        if n_rows * n_per_row < len(imgs):
            n_rows += 1
        n_cols = min(len(imgs), n_per_row)
        H, W = imgs[0].shape[:2]
        grid_img = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            r = i // n_per_row
            c = i % n_per_row
            grid_img[r * H : (r + 1) * H, c * W : (c + 1) * W] = img
        # Rescale image
        if output_width is not None:
            output_height = int(
                output_width * grid_img.shape[0] / grid_img.shape[1]
            )
            grid_img = cv2.resize(grid_img, (output_width, output_height))
        return grid_img

    @staticmethod
    def add_title_to_image(
        img: np.ndarray, title: str, font_size: int = 50, bg_color=200,
        fg_color=(0, 0, 255)
    ):
        font_img = np.zeros((font_size, img.shape[1], 3), dtype=np.uint8)
        font_img.fill(bg_color)
        font_img = Image.fromarray(font_img)
        draw = ImageDraw.Draw(font_img)
        # Find a font file
        mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
        file = font_manager.findfont(mpl_font)
        font = ImageFont.truetype(font=file, size=25)
        draw.text((20, 5), title, fg_color, font=font)
        font_img = np.array(font_img)
        return np.concatenate([font_img, img], axis=0)


class SemanticMapPrecomputedDataset(SemanticMapDataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.dset = cfg.dset_name
        # Seed the dataset
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        # Load map paths
        all_map_paths = sorted(
            glob.glob(osp.join(cfg.root, split, f"**/*.pbz2"), recursive=True)
        )
        # print("all_map_paths=",all_map_paths)
        
        map_paths = [path for path in all_map_paths if not path.endswith("_room.pbz2")]

        self.map_paths = map_paths
        # print("map_path1=",map_paths)
        # Both locations and directions cannot be enabled at the same time.
        # assert not (self.cfg.enable_locations and self.cfg.enable_directions)

    def __len__(self):
        return len(self.map_paths)

    def compute_object_pfs(self, fmm_dists):
        cutoff = self.cfg.object_pf_cutoff_dist
        opfs = torch.clamp((cutoff - fmm_dists) / cutoff, 0.0, 1.0)
        # cutoff= 5.0
        # fmm distance is each frontier to target object
        # print("in compute_object_pf functions!!")
        # print("fmm_dists=",fmm_dists.shape)
        # print("cutoff=",cutoff)
        # print("opfs=",opfs.shape)
        return opfs

    def __getitem__(self, idx):
        # print("idx in getitem=",idx)
        # print("map_path2=",self.map_paths)


        global obj_room_sc
        # print("obj_room_sc in getitem=",obj_room_sc.shape,obj_room_sc[0])

        with bz2.BZ2File(self.map_paths[idx], 'rb') as fp:
            print("self.map_paths[idx]=",self.map_paths[idx])
            data = cPickle.load(fp)

        world_xyz = data["world_xyz"] 
        heading = data["world_heading"]

        print("data[world_xyz]=",world_xyz)
        print("data[heading]=",heading)

        # load same scene room pbz2 file
        # 获取文件路径的目录和文件名部分
        directory, filename = os.path.split(self.map_paths[idx])
        # print("directory, filename=",directory,filename)
        # directory = directory + "_room"

        new_filename = filename.replace(".pbz2", "_room.pbz2")
        # print("new_filename=", new_filename)
       

        # 重新组合新的文件路径
        map_paths_room = os.path.join(directory, new_filename)

        with bz2.BZ2File(map_paths_room, 'rb') as fp_room:
            # print("map_paths_room=",map_paths_room)
            data_room = cPickle.load(fp_room)

        world_xyz_room = data_room["world_xyz"] 
        heading_room = data_room["world_heading"]

        print("data_room[world_xyz]=",world_xyz_room)
        print("data_room[heading]=",heading_room)


        # Convert cm -> m
        data['fmm_dists'] = data['fmm_dists'].astype(np.float32) / 100.0
        in_semmap = torch.from_numpy(data['in_semmap'])
        in_semmap_room = torch.from_numpy(data_room['in_semmap']) 


        semmap = torch.from_numpy(data['semmap'])
        semmap_room = torch.from_numpy(data_room['semmap'])



        fmm_dists = torch.from_numpy(data['fmm_dists'])
        # Compute object_pfs
        object_pfs = self.compute_object_pfs(fmm_dists)
        loss_masks, masks, dirs, locs, area_pfs, acts, frontiers = self.get_masks_and_labels(
            in_semmap, semmap, fmm_dists
        )
        # print("loss_masks=",loss_masks.shape)
        # print("locs=",locs)
        # print("acts=",acts)
        # plt.imshow(loss_masks[3])
        # plt.savefig("loss_masks[3].png")
        # reference_layer = loss_masks[0]

        # # 遍历其余的层并与参考层进行比较
        # for i in range(1, loss_masks.shape[0]):
        #     if not (loss_masks[i] == reference_layer).all():
        #         print(f"Layer {i} is different from the reference layer!")
        #     else:
        #         print("all same")






        if self.cfg.potential_function_masking:
            # print(" self.cfg.potential_function_masking=", self.cfg.potential_function_masking)
            object_pfs = torch.clamp(object_pfs * masks, 0.0, 1.0)
            # print("obj_pfs after mask =",object_pfs.shape)


        # plt.imshow(object_pfs[2])
        # plt.savefig("object_pfs[2]_.png")  
        # plt.imshow(masks[2])
        # plt.savefig("masks[2]_.png") 


        
        # calculate room pf 
         
        # print("object_pfs!!!!!!!!!!!!!=",object_pfs.shape)
        # plt.imshow(object_pfs[7])
        # plt.savefig("object_pfs_[7]!!!!!!!!!!!!!!.png")

        obj_room_sc_map = torch.from_numpy(np.zeros((480, 480)))
        obj_room_sc_maps = torch.from_numpy(np.zeros((15,480,480)))
        result_obj_room_maps = torch.from_numpy(np.zeros((17,480,480)))


        for m in range (0,15):
            obj_room_sc_map.fill_(0)
            # print("m=",m) m=0-14
            for i in range(2, 21):
                mask = semmap_room[i] == 1
                obj_room_sc_map = torch.add(obj_room_sc_map, mask * obj_room_sc[m,i-2])
                # print("obj_room_sc_map=",obj_room_sc_map.shape)
            # plt.imshow(obj_room_sc_map,cmap='YlGnBu', origin='upper')
            # plt.colorbar(label='Room-object relation Value')
            # plt.savefig('obj_room_sc_map_1.png', dpi=300)

                frontier_mask = object_pfs[0] > 0
                obj_room_sc_frontier_map = torch.from_numpy( np.where(frontier_mask,obj_room_sc_map,0))
                # print("obj_room_sc_frontier+map=",obj_room_sc_frontier_map.shape)
        
                # plt.imshow(obj_room_sc_frontier_map,cmap='YlGnBu', origin='upper')
                # plt.colorbar(label='Room-object relation Value')
                # plt.savefig('obj_room_sc_frontier_map_14.png', dpi=300)
            obj_room_sc_maps[m] = obj_room_sc_frontier_map
        print("obj_room_sc_maps shape ==",obj_room_sc_maps.shape)

        result_obj_room_maps[:2] = object_pfs[:2]
        result_obj_room_maps[2:] = obj_room_sc_maps

            
        # plt.imshow(obj_room_sc_maps[14],cmap='YlGnBu',  vmin=-1, vmax=1, origin='upper')
        # plt.colorbar()
        # # plt.title('Partial object-room frontier map (couch)')
        # plt.savefig('ZZZZZZobj_room_sc_maps_couch.png', dpi=300)

    
        # plt.imshow(result_obj_room_maps[2],cmap='YlGnBu',   vmin=-1, vmax=1,origin='upper')
        # plt.colorbar()
        # plt.savefig('zzzzzresult_obj_room_maps[0].png', dpi=300)

        # print("result_obj_room_maps in getitem=",result_obj_room_maps.shape)




        input = {'semmap': in_semmap}
        input_room ={'semmap': in_semmap_room}



        ########################################################################
        # Optimizations for reducing memory usage during data-loading
        ########################################################################
        # Convert object_pfs to integers (0 -> 1000)
        object_pfs = (object_pfs * 1000.0).int()

        result_obj_room_maps = (result_obj_room_maps*1000.0).int()

        label = {
            'semmap': semmap,
            'object_pfs': object_pfs,
            'loss_masks': loss_masks,
        }
        label_room = {
            'semmap': semmap_room,
            'object_pfs' : result_obj_room_maps,
            'loss_masks' : loss_masks,
        }



        # Convert area-pfs to integers (0 -> 1000)
        if area_pfs is not None:
            area_pfs = (area_pfs * 1000.0).int()
        ########################################################################

        if dirs is not None:
            label['dirs'] = dirs
            label_room['dirs'] =dirs
        if locs is not None:
            label['locs'] = locs
            label_room['locs'] = locs
        if area_pfs is not None:
            label['area_pfs'] = area_pfs
            label_room['area_pfs'] = area_pfs
        if acts is not None:
            label['acts'] = acts
            label_room['acts'] = acts
        if frontiers is not None:
            label['frontiers'] = frontiers
            label_room['frontiers'] = frontiers


    
        # Free memory
        del data
        gc.collect()
        print("before return!!!!")
        return input, label, input_room, label_room, world_xyz, heading

    def get_masks_and_labels(self, in_semmap, out_semmap, out_fmm_dists):
        # Expand to add batch dim
        in_semmap = in_semmap.unsqueeze(0)
        out_semmap = out_semmap.unsqueeze(0)
        out_fmm_dists = out_fmm_dists.unsqueeze(0)
        N = in_semmap.shape[1]
        # Create a scaling-mask for the loss function / potential field
        # By default, select only navigable/object regions where fmm dist exists
        out_base_masks = torch.any(out_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        out_base_masks = repeat(out_base_masks, '() () h w -> () n h w', n=N).float()
        ################### Build mask based on input regions ##################
        # Compute an advanced mask based on input regions.
        out_masks = torch.any(out_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        out_masks = repeat(out_masks, '() () h w -> () n h w', n=N).float()
        # Compute frontier locations
        free_map = in_semmap[0, FLOOR_ID] # (H, W)
        # Dilate the free map
        if self.cfg.dilate_free_map:
            free_map = free_map.float().unsqueeze(0).unsqueeze(1)
            for i in range(self.cfg.dilate_iters):
                free_map = torch.nn.functional.max_pool2d(
                    free_map, 7, stride=1, padding=3
                )
            free_map = free_map.bool().squeeze(1).squeeze(0)
        exp_map = torch.any(in_semmap, dim=1)[0] # (H, W)
        exp_map = exp_map | free_map
        unk_map = ~exp_map
        unk_map = unk_map.numpy()
        free_map = free_map.numpy()
        frontiers = get_frontiers_np(unk_map, free_map) # (H, W)
        # Compute contours of frontiers
        contours = None
        if self.cfg.enable_unexp_area:
            contours, _ = cv2.findContours(
                frontiers.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [contour[:, 0].tolist() for contour in contours] # Clean format
        frontiers = torch.from_numpy(frontiers).unsqueeze(0).unsqueeze(1)
        # Dilate the frontiers mask
        frontiers_mask = torch.nn.functional.max_pool2d(
            frontiers.float(), 7, stride=1, padding=3
        ).bool() # (1, N or 1, H, W)
        # Scaling at the frontiers
        alpha = self.cfg.potential_function_frontier_scaling
        # Scaling at the non-visible regions
        beta = self.cfg.potential_function_non_visible_scaling
        visibility_mask = torch.any(in_semmap, dim=1, keepdim=True) # (1, 1, H, W)
        # Scaling at the visible & non-frontier regions
        gamma = self.cfg.potential_function_non_frontier_scaling
        not_frontier_or_visible = ~(visibility_mask | frontiers_mask)
        visible_and_not_frontier = visibility_mask & (~frontiers_mask)
        # Compute final mask
        out_masks = out_masks * (
            visible_and_not_frontier * gamma + \
            not_frontier_or_visible * beta + \
            frontiers_mask * alpha
        )
        # Compute directions to each object from map center if needed
        ## For each category, pick the object nearest (euclidean distance)
        ## to the center.  Them compute the directions from center to
        ## this object. Conventions: East is 0, clockwise is positive
        out_dirs = None
        if self.cfg.enable_directions:
            out_dirs = []
            all_dirs = np.array(self.cfg.prediction_directions)
            ndir = len(self.cfg.prediction_directions)
            for sem_map in out_semmap[0]: # (H, W)
                sem_map = sem_map.cpu().numpy()
                H, W = sem_map.shape
                Hby2, Wby2 = H // 2, W // 2
                # Discover connected components (i.e., object instances)
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    sem_map.astype(np.uint8) * 255 , 4 , cv2.CV_32S
                )
                # Ignore 1st element of centroid since it's the image center
                centroids = centroids[1:]
                if len(centroids) == 0:
                    # class N is object missing class
                    out_dirs.append(ndir)
                    continue
                map_y, map_x = centroids[:, 1], centroids[:, 0]
                # Pick closest instance of the object
                dists = np.sqrt((map_y - Hby2) ** 2 + (map_x - Wby2) ** 2)
                min_idx = np.argmin(dists)
                obj_y, obj_x = map_y[min_idx], map_x[min_idx]
                obj_dir = np.arctan2(obj_y - Hby2, obj_x - Wby2)
                obj_dir = (np.rad2deg(obj_dir) + 360.0) % 360.0
                # Classify obj_dir into [0, ..., ndir-1] classes
                dir_cls = np.argmin(np.abs(all_dirs - obj_dir))
                out_dirs.append(dir_cls)
            out_dirs = torch.LongTensor(out_dirs).to(out_masks.device) # (N, )
        # Compute position to each object from map center if needed
        ## For each category, pick the object nearest (euclidean distance) to the center
        ## The compute the central position of the object in this map.
        ## Normalize the position b/w 0 to 1. Output is (x, y).
        ## Conventions: East is X, South is Y, map top-left is (0, 0)
        out_locs = None
        if self.cfg.enable_locations:
            out_locs = []
            for sem_map in out_semmap[0]: # (H, W)
                sem_map = sem_map.cpu().numpy()
                H, W = sem_map.shape
                Hby2, Wby2 = H // 2, W // 2
                # Discover connected components (i.e., object instances)
                _, _, _, centroids = cv2.connectedComponentsWithStats(
                    sem_map.astype(np.uint8) * 255 , 4 , cv2.CV_32S
                )
                # Ignore 1st element of centroid since it's the image center
                centroids = centroids[1:]
                if len(centroids) == 0:
                    out_locs.append((-1, -1))
                    continue
                map_y, map_x = centroids[:, 1], centroids[:, 0]
                # Pick closest instance of the object
                dists = np.sqrt((map_y - Hby2) ** 2 + (map_x - Wby2) ** 2)
                min_idx = np.argmin(dists)
                obj_y, obj_x = map_y[min_idx], map_x[min_idx]
                # Normalize this to (0, 1) range
                obj_y = obj_y / H
                obj_x = obj_x / W
                out_locs.append((obj_x, obj_y))
            out_locs = torch.Tensor(out_locs).to(out_masks.device) # (N, 2)
        # Compute action needed to reach each object from map center if needed.
        ## Assume that the agent is at the map center, facing right.
        out_acts = None
        if hasattr(self.cfg, 'enable_actions') and self.cfg.enable_actions:
            out_acts = []
            traversible = out_semmap[0, 0] | (~torch.any(out_semmap[0], dim=0)) # (H, W)
            planner = FMMPlanner(traversible.float().cpu().numpy())
            H, W = traversible.shape
            Hby2, Wby2 = H // 2, W // 2
            traversible[Hby2 - 3:Hby2 + 4, Wby2 - 3:Wby2 + 4] = 1
            for i, (sem_map, fmm_dist) in enumerate(zip(out_semmap[0], out_fmm_dists[0])): # (H, W)
                sem_map = sem_map.cpu().numpy()
                # Use pre-computed fmm dists
                fmm_dist = fmm_dist.cpu().numpy()
                H, W = sem_map.shape
                assert H == W
                map_resolution = self.cfg.output_map_size / H
                if not np.any(sem_map > 0):
                    out_acts.append(-1)
                    continue
                # planner.fmm_dist = np.floor(fmm_dist / map_resolution)
                goal_map = sem_map.astype(np.float32)
                selem = skmp.disk(0.5 / map_resolution)
                goal_map = skmp.binary_dilation(goal_map, selem) != True
                goal_map = 1 - goal_map * 1.
                planner.set_multi_goal(goal_map)

                start = (H // 2, W // 2)
                stg_x, stg_y, _, stop = planner.get_short_term_goal(start)
                if stop:
                    out_acts.append(0) # STOP
                else:
                    angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                            stg_y - start[1]))
                    angle_agent = 0.0
                    relative_angle = (angle_agent - angle_st_goal) % 360.0
                    if relative_angle > 180:
                        relative_angle -= 360
                    
                    if relative_angle > self.cfg.turn_angle / 2.0:
                        out_acts.append(3) # TURN-RIGHT
                    elif relative_angle < -self.cfg.turn_angle / 2.0:
                        out_acts.append(2) # TURN-RIGHT
                    else:
                        out_acts.append(1) # MOVE-FORWARD
            out_acts = torch.Tensor(out_acts).long().to(out_masks.device) # (N,)

        # Compute unexplored free-space starting from each frontier
        out_area_pfs = None
        if self.cfg.enable_unexp_area:
            floor_map = out_semmap[0, FLOOR_ID] # (H, W)
            # print("floor_map=",floor_map.shape)
            image = np.array(floor_map)
            image = image*255
            ts = calendar.timegm(time.gmtime())

            # cv2.imwrite(str(dir) + "floor_map_"+ str(ts) +".png",image)
            unexp_map = ~torch.any(in_semmap[0], dim=0) # (H, W)
            # print("unexp_map=",unexp_map.shape)
            image1 = np.array(unexp_map)
            image1 = image1*255
            # cv2.imwrite(str(dir)+"unexp_map_"+str(ts)+".png",image1)

            unexp_floor_map = floor_map & unexp_map # (H, W)

            image2 = np.array(unexp_floor_map)
            image2 = image2*255
            # cv2.imwrite(str(dir)+"unexp_floor_map_"+str(ts)+".png",image2)

            # Identify connected components of unexplored floor space
            unexp_floor_map = unexp_floor_map.cpu().numpy()
            unexp_floor_map = unexp_floor_map.astype(np.uint8) * 255
            ncomps, comp_labs, _, _ = cv2.connectedComponentsWithStats(
                unexp_floor_map, 4 , cv2.CV_32S
            )
            # print("ncomps=",ncomps)
            # Only select largest 5 contours
            largest_contours = sorted(
                contours, key=lambda cnt: len(cnt), reverse=True
            )[:5]
            contour_stats = [0.0 for _ in range(len(largest_contours))]
            # For each connected component, find the intersecting frontiers and
            # add area to them.
            kernel = np.ones((5, 5))
            for i in range(1, ncomps):
                comp = (comp_labs == i).astype(np.float32)
                # print("comp=",comp)
                image3 = np.array(comp)
                image3 = image3*255
                # cv2.imwrite(str(dir)+"comp_"+str(ts)+".png",image3)
                comp_area = comp.sum().item() * (self.grid_size ** 2)
                # print("comp_area=",comp_area)
                # dilate
                comp = cv2.dilate(comp, kernel, iterations=1)
                image4 = np.array(comp)
                image4 = image4*255
                # cv2.imwrite(str(dir)+"compdilate_"+str(ts)+".png",image4)
                comp_area = comp.sum().item() * (self.grid_size ** 2)
                # intersect with frontiers
                for j, contour in enumerate(largest_contours):
                    intersection = 0.0
                    for x, y in contour:
                        intersection += comp[y, x]
                    if intersection > 0:
                        contour_stats[j] += comp_area
            # Create out areas map
            out_area_pfs = torch.zeros_like(floor_map).float() # (H, W)
            if hasattr(self.cfg, 'normalize_area_by_constant'):
                normalize_area_by_constant = self.cfg.normalize_area_by_constant
            else:
                normalize_area_by_constant = False

            if normalize_area_by_constant:
                total_area = self.cfg.max_unexp_area
            else:
                total_area = floor_map.sum().item() * (self.grid_size ** 2) / 2.0
            for stat, contour in zip(contour_stats, largest_contours):
                # Use linear scoring
                score = np.clip(stat / (total_area + EPS), 0.0, 1.0)
                for x, y in contour:
                    out_area_pfs[y, x] = score
            # Dilate the area map
            image5 = np.array(out_area_pfs)
            image5 = image5*255
            # cv2.imwrite(str(dir)+"out_area_pfs_"+str(ts)+".png",image5)
            # print("out_area_pf_before dilate=",out_area_pfs.shape)

            out_area_pfs = out_area_pfs.unsqueeze(0).unsqueeze(1) # (1, 1, H, W)
            out_area_pfs = torch.nn.functional.max_pool2d(
                out_area_pfs, 7, stride=1, padding=3
            )
            # print("out_area_pfs after dilate=",out_area_pfs.shape)
            out_area_pfs = out_area_pfs.squeeze(1) # (1, H, W)
            # print("out_area_pfs after squeeze=",out_area_pfs.shape)
            tmp_out_area_pfs = out_area_pfs.squeeze(0)
            # print("tmp_out_area_pfs=",tmp_out_area_pfs.shape)
            image6 = np.array(tmp_out_area_pfs)
            image6 = image6*255
            # cv2.imwrite(str(dir)+"tmp_out_area_pfs_"+str(ts)+".png",image6)

        # Remove batch dim
        out_base_masks = out_base_masks.squeeze(0)
        out_masks = out_masks.squeeze(0)

        # print("out_dirs, out_locs, out_acts=",out_dirs, out_locs,out_acts)
        return out_base_masks, out_masks, out_dirs, out_locs, out_area_pfs, out_acts, contours
