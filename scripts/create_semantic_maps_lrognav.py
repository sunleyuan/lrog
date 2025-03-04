import glob
import json
import math
import multiprocessing as mp
import os
import random
import re

from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
import tqdm
import trimesh
import pywavefront

from PIL import Image, ImageDraw, ImageFont
from torch_scatter import scatter_max

Image.MAX_IMAGE_PIXELS = 1000000000
import poni.hab_utils as hab_utils
from matplotlib import font_manager
from plyfile import PlyData

from poni.constants import d3_40_colors_rgb, OBJECT_CATEGORIES, SPLIT_SCENES
random.seed(123)

room_floor_cat = None 


################################################################################
# Gibson constants
################################################################################
GIBSON_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["gibson"]
GIBSON_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(GIBSON_CATEGORIES)}
# print("GIBSON_CATEGORY_MAP=",GIBSON_CATEGORY_MAP)
# 这个颜色是从chair开始的，前面还有三个 out-of-bonds, floor 和 wall
GIBSON_OBJECT_COLORS = [
    (0.9400000000000001, 0.7818, 0.66),
    (0.9400000000000001, 0.8868, 0.66),
    (0.8882000000000001, 0.9400000000000001, 0.66),
    (0.7832000000000001, 0.9400000000000001, 0.66),
    (0.6782000000000001, 0.9400000000000001, 0.66),
    (0.66, 0.9400000000000001, 0.7468000000000001),
    (0.66, 0.9400000000000001, 0.8518000000000001),
    (0.66, 0.9232, 0.9400000000000001),
    (0.66, 0.8182, 0.9400000000000001),
    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),
]





# 17 categories include floor and wall
# OBJECT_CATEGORIES = {
#     "gibson": [
#         "floor",
#         "wall",
#         "chair",
#         "couch",
#         "potted plant",
#         "bed",
#         "toilet",
#         "tv",
#         "dining-table",
#         "oven",
#         "sink",
#         "refrigerator",
#         "book",
#         "clock",
#         "vase",
#         "cup",
#         "bottle",
#     ]






#21 categories include floor and wall
ROOM_CATEGORIES = [
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
    
GIBSON_ROOM_CATEGORIES =  ["out-of-bounds"]+ ROOM_CATEGORIES
GIBSON_ROOM_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(GIBSON_ROOM_CATEGORIES)}
# print("GIBSON_ROOM_CATEGORY_MAP=",GIBSON_ROOM_CATEGORY_MAP)
GIBSON_ROOM_COLORS = [
 (1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0), (1.0, 1.0, 0.5), (1.0, 0.5, 1.0),
    (0.5, 1.0, 1.0), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5), (1.0, 0.75, 0.5),
    (0.75, 1.0, 0.5), (0.5, 0.75, 1.0), (1.0, 0.5, 0.75), (0.75, 0.5, 1.0), (0.5, 1.0, 0.75),
    (0.7, 0.7, 0.7), (0.3, 0.3, 0.3), (0.9, 0.75, 0.55), (0.55, 0.75, 0.9)
]
################################################################################
# MP3D constants
################################################################################
MP3D_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["mp3d"]
MP3D_CATEGORY_MAP = {obj: idx for idx, obj in enumerate(MP3D_CATEGORIES)}
MP3D_OBJECT_COLORS = []  # Excludes 'out-of-bounds', 'floor', and 'wall'
for color in d3_40_colors_rgb[: len(MP3D_CATEGORIES) - 3]:
    color = (color.astype(np.float32) / 255.0).tolist()
    MP3D_OBJECT_COLORS.append(color)

################################################################################
# General constants
################################################################################
# assert "ACTIVE_DATASET" in os.environ
# ACTIVE_DATASET = os.environ["ACTIVE_DATASET"]  # mp3d / gibson

ACTIVE_DATASET = "gibson"



if ACTIVE_DATASET == "mp3d":
    OBJECT_COLORS = MP3D_OBJECT_COLORS
    OBJECT_CATEGORIES = MP3D_CATEGORIES
    OBJECT_CATEGORY_MAP = MP3D_CATEGORY_MAP
    SCENES_ROOT = "data/scene_datasets/mp3d_uncompressed"
    SB_SAVE_ROOT = "data/semantic_maps/mp3d/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/mp3d/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/mp3d/semantic_maps"
    NUM_WORKERS = 8
    MAX_TASKS_PER_CHILD = 2
    SAMPLING_RESOLUTION = 0.20
    WALL_THRESH = [0.25, 1.25]
else:
    OBJECT_COLORS = GIBSON_OBJECT_COLORS
    OBJECT_CATEGORIES = GIBSON_CATEGORIES
    OBJECT_CATEGORY_MAP = GIBSON_CATEGORY_MAP
    SCENES_ROOT = "data/scene_datasets/gibson_semantic"
    SB_SAVE_ROOT = "data/semantic_maps/gibson/scene_boundaries"
    PC_SAVE_ROOT = "data/semantic_maps/gibson/point_clouds"
    SEM_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps"
    NUM_WORKERS = 12
    MAX_TASKS_PER_CHILD = None
    SAMPLING_RESOLUTION = 0.10
    WALL_THRESH = [0.25, 1.25]

COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]
LEGEND_PALETTE = [
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *OBJECT_COLORS,
]

COLOR_PALETTE_ROOM = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in GIBSON_ROOM_COLORS for oci in oc],
]
LEGEND_PALETTE_ROOM =[
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *GIBSON_ROOM_COLORS,
]
 


def get_palette_image():
    # Find a font file
    mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(mpl_font)
    font = ImageFont.truetype(font=file, size=20)

    # Save color palette
    cat_size = 30
    buf_size = 10
    text_width = 150

    image = np.zeros(
        (cat_size * len(OBJECT_CATEGORIES), cat_size + buf_size + text_width, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for i, (category, color) in enumerate(zip(OBJECT_CATEGORIES, LEGEND_PALETTE)):
        color = tuple([int(c * 255) for c in color])
        draw.rectangle(
            [(0, i * cat_size), (cat_size, (i + 1) * cat_size)],
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            [cat_size + buf_size, i * cat_size],
            category,
            font=font,
            fill=(255, 255, 255),
        )
    print("SAVE PALTTE IMAGE")
    # cv2.imwrite("palette_image.png",np.array(image))
    return np.array(image)



def get_palette_image_room():
    # Find a font file
    mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(mpl_font)
    font = ImageFont.truetype(font=file, size=16)

    # Save color palette
    cat_size = 30
    buf_size = 10
    text_width = 150

    image = np.zeros(
        (cat_size * len(GIBSON_ROOM_CATEGORIES), cat_size + buf_size + text_width, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for i, (category, color) in enumerate(zip(GIBSON_ROOM_CATEGORIES, LEGEND_PALETTE_ROOM)):
        color = tuple([int(c * 255) for c in color])
        draw.rectangle(
            [(0, i * cat_size), (cat_size, (i + 1) * cat_size)],
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            [cat_size + buf_size, i * cat_size+5],
            category,
            font=font,
            fill=(255, 255, 255),
        )
    print("SAVE PALTTE IMAGE")
    # cv2.imwrite("palette_image.png",np.array(image))
    return np.array(image)



def extract_scene_point_clouds(
    glb_path,
    ply_path,
    scn_path,
    houses_dim_path,
    pc_save_path,
    sampling_density=1600.0,
):
    # global room_floor_cat
    print(f"Scene: {glb_path}")
    # Get mapping from object instance id to category
    if os.path.isfile(scn_path):
        with open(scn_path) as fp:
            # print("scn_path fro scn_data=",scn_path)
            scn_data = json.load(fp)
            # print("scn_data=",scn_data)
        obj_id_to_cat = {
            obj["id"]: obj["class_"]
            for obj in scn_data["objects"]
            if obj["class_"] in OBJECT_CATEGORY_MAP
        } ; 
        # print("obj_id_to_cat=",obj_id_to_cat);
        # extract room category
        room_id_to_cat = {
            room["id"]: room["scene_category"]
            for room in scn_data["rooms"]
            if room["scene_category"] in GIBSON_ROOM_CATEGORY_MAP
        } ; 
        # print("room_id_to_cat=",room_id_to_cat)
        room_floor_to_cat = {
            room["id"]: room["floor_number"]
            for room in scn_data["rooms"]
            if room["scene_category"] in GIBSON_ROOM_CATEGORY_MAP
        };
        # print("room_floor_to_cat=",room_floor_to_cat)
        # room_floor_cat = room_floor_to_cat
        # print("room_floor_cat=",room_floor_cat)
    else:
        sim = hab_utils.robust_load_sim(glb_path)
        objects = sim.semantic_scene.objects
        # print("objects=",objects)
        obj_id_to_cat = {}
        for obj in objects:
            obj_id = obj.id.split("_")[-1]  # <level_id>_<region_id>_<object_id>
            obj_cat = obj.category.name()
            if obj_cat not in OBJECT_CATEGORY_MAP or obj_cat in ["wall", "floor"]:
                continue
            obj_id_to_cat[int(obj_id)] = obj_cat
        sim.close()
    ############################################################################
    # Get vertices for all objects
    ############################################################################
    vertices = []
    colors = []
    obj_ids = []
    sem_ids = []
    ply_data = PlyData.read(ply_path)
    # print("ply_path=",ply_path)
    # print("ply_data=",ply_data)
    # Get faces for each object id
    obj_id_to_faces = defaultdict(list)
    # print("obj_id_to_faces list=",list)
    # print("obj_id_to_faces=",obj_id_to_faces)
    for face in ply_data["face"]:
        # face= (array([14549, 14434, 13602], dtype=int32), 0)
        # ply_data[face]= element face 1624279
        # print("face=",face)
        # print("ply_data[face]=",ply_data["face"])
        vids = list(face[0])
        obj_id = face[1]
        # print("len(face)=",len(face))
        # print("vids=",vids)
        # print("obj_id=",obj_id)
        if obj_id in obj_id_to_cat:
            # print("obj_id_to_cat=",obj_id_to_cat)
            # print("obj_id=",obj_id)
            p1 = ply_data["vertex"][vids[0]]
            # print("p1=",p1)
            p1 = [p1[0], p1[2], -p1[1]]
            # print("p1=",p1)
            # p1= (1.12565, -3.7905295, 3.6241899, 128, 128, 128)
            # p1= [1.12565, 3.6241899, 3.7905295]
            p2 = ply_data["vertex"][vids[1]]
            p2 = [p2[0], p2[2], -p2[1]]
            p3 = ply_data["vertex"][vids[2]]
            p3 = [p3[0], p3[2], -p3[1]]
            obj_id_to_faces[obj_id].append([p1, p2, p3])
    # print("obj_id_to_faces[107]=",obj_id_to_faces[107])        
    # Get dense point-clouds for each object id
    for obj_id, faces in obj_id_to_faces.items():
        # print("obj_id_to_faces.items()=",obj_id_to_faces.items())
        ocat = obj_id_to_cat[obj_id]
        # print("ocat=",ocat)
#         ocat= book
# ocat= book
# ocat= book
# ocat= sink
# ocat= oven
# ocat= potted plant
# ocat= sink
# ocat= potted plant
# ocat= potted plant
# ocat= book
# ocat= bed
        sem_id = OBJECT_CATEGORY_MAP[ocat]
        # print("sem_id=",sem_id)
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        # print("color=",color)
        # Create trimesh vertices and faces from faces
        faces = np.array(faces)  # (N, 3, 3)
        # print("faces=",faces.shape)
        t_pts = hab_utils.dense_sampling_trimesh(faces, sampling_density)
        # print("t_pts=",t_pts)
        for t_pt in t_pts:
            vertices.append(t_pt)
            # print("vertices=",vertices)
            obj_ids.append(obj_id)
            sem_ids.append(sem_id)
            colors.append(color)
    # print("vertices=",vertices)
    # print("colors=",colors)
    # print("obj_ids=",obj_ids)


    ############################################################################
    # Get vertices for all rooms
    ############################################################################
    vertices_room =[]
    colors_room = []
    room_ids = []
    sem_room_ids = []
    room_floor = []
    obj_data = pywavefront.Wavefront('/home/aae14859ln/Sun/PONI/visuals_room/Darden/automated_graph/Darden_room_inst.obj',create_materials=True,
    collect_faces=True)
    # print("obj_data=",obj_data)
    # print(obj_data.vertices[578354])   
    # print(obj_data.vertices[647261])  
    # print(obj_data.vertices[138531])  
    # print(obj_data.materials['bathroom'].vertices[0:9])
    # print(obj_data.vertices[249972])
    # print(obj_data.vertices[478427])
    # print(obj_data.vertices[391105])
    #  [[-6.20963, 1.77441, 0.615234], [-6.20902, 1.77215, 0.616229], [-6.21061, 1.77832, 0.652018]]]]
    # the number of vertices in faces for each material should -1
    # e.g.  g Mesh
    # usemtl bathroom
    # f 578355 647262 138532 but obj_data.vertices[578355-1] = obj_data.materials['bathroom'].vertices[0:3]
    # print(obj_data.mesh_list[0].faces[0]) #[578354, 647261, 138531]
    # for name, material in obj_data.materials.items():
        # print(name)
    # print(obj_data.mateials.items())
    room_id_to_faces = defaultdict(list)
    # print("room_id_to_faces=",room_id_to_faces)


    room_id_to_cat_wnum= {}
    for key, value in room_id_to_cat.items():
        new_value = f"{value}_{key}"
        room_id_to_cat_wnum[key] = new_value

    # print("room_id_to_cat_wnum=",room_id_to_cat_wnum)



    for room_id in room_id_to_cat_wnum:
        # print("room_id111111=",room_id)
        # print(room_id_to_cat[room_id])
        room_name = room_id_to_cat_wnum[room_id]
        # print("room_name=",room_name)
        p_room = obj_data.materials[room_name].vertices[:]
        # print(p_room)
        p_room_ = [p_room[i:i+3] for i in range(0,len(p_room),3)]
        p_room_ha = [sub if len(sub) < 3 else [sub[0], sub[1], sub[2]] for sub in p_room_]
        p_room__ = [p_room_ha[i:i+3] for i in range(0,len(p_room_ha),3)]
        room_id_to_faces[room_id].append(p_room__)





    # print(np.array(room_id_to_faces[1]).squeeze(axis=0)[0])
    # print(np.array(room_id_to_faces[2]).squeeze(axis=0)[0])
    # print(np.array(room_id_to_faces[3]).squeeze(axis=0)[0])
    for room_id, faces_room in room_id_to_faces.items():
        # print("room_id222222=",room_id)
        # floor_number = room_floor_to_cat[room_id]
        # print("floor_number=",floor_number)
        # if floor_number == 'A':
        #     floor_number_ = 0
        # if floor_number == 'B':
        #     floor_number_ = 1
        # if floor_number == 'C':
        #     floor_number_ = 2
        # if floor_number == 'D':
        #     floor_number_ = 3
        # if floor_number == 'E':
        #     floor_number_ = 4
        # print("floor_number_ =",floor_number_)
        rcat = room_id_to_cat[room_id]
        # print("rcat=",rcat)
        sem_room_id = GIBSON_ROOM_CATEGORY_MAP[rcat]
        # print("sem_room_id=",sem_room_id)
        room_color = GIBSON_ROOM_COLORS[sem_room_id-3]
        # print("room_color=",room_color)
        room_faces = np.array(faces_room)
        # print("room_faces=",room_faces)
        # room_faces2 = np.array(room_id_to_faces[room_id])
        room_faces = room_faces.squeeze(axis=0)
        # print("room_faces=",room_faces.shape)
        room_t_pts = hab_utils.dense_sampling_trimesh(room_faces,sampling_density)
        for t_pt in room_t_pts:
            vertices_room.append(t_pt)
            room_ids.append(room_id)
            sem_room_ids.append(sem_room_id)
            colors_room.append(room_color)
            # room_floor.append(floor_number_)


    ############################################################################
    # Get vertices for navigable spaces
    ############################################################################
    sim = hab_utils.robust_load_sim(glb_path)
    # print("glb_path=",glb_path)
    navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
    t_pts = hab_utils.dense_sampling_trimesh(navmesh_triangles, sampling_density)
    for t_pt in t_pts:
        obj_id = -1
        sem_id = OBJECT_CATEGORY_MAP["floor"]
        # print("sem_id floor=",sem_id)
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        # print("floor color=",color)
        vertices.append(t_pt)
        obj_ids.append(obj_id)
        sem_ids.append(sem_id)
        colors.append(color)
        
        vertices_room.append(t_pt)
        room_ids.append(obj_id)
        sem_room_ids.append(sem_id)
        colors_room.append(color)
    sim.close()

    ############################################################################
    # Get vertices for walls
    ############################################################################
    per_floor_wall_pc = extract_wall_point_clouds(
        glb_path, houses_dim_path, sampling_density=sampling_density
    )
    for _, points in per_floor_wall_pc.items():
        obj_id = -1
        sem_id = OBJECT_CATEGORY_MAP["wall"]
        # print("sem id walls=",sem_id)
        color = COLOR_PALETTE[sem_id * 3 : (sem_id + 1) * 3]
        # print("color wall =",color)
        for p in points:
            vertices.append(p)
            obj_ids.append(obj_id)
            sem_ids.append(sem_id)
            colors.append(color)

            vertices_room.append(p)
            room_ids.append(obj_id)
            sem_room_ids.append(sem_id)
            colors_room.append(color)

    vertices = np.array(vertices)
    obj_ids = np.array(obj_ids)
    sem_ids = np.array(sem_ids)
    colors = np.array(colors)

    # print("pc_save_path=",pc_save_path)

    with h5py.File(pc_save_path, "w") as fp:
        fp.create_dataset("vertices", data=vertices)
        fp.create_dataset("obj_ids", data=obj_ids)
        fp.create_dataset("sem_ids", data=sem_ids)
        fp.create_dataset("colors", data=colors)

########
# save H5 file for room segmentation
#######

    vertices_room = np.array(vertices_room)
    room_ids = np.array(room_ids)
    sem_room_ids = np.array(sem_room_ids)
    colors_room = np.array(colors_room)
    floor_number = np.array(room_floor)
    # room_floor_to_cat=np.array(room_floor_to_cat)

    letters = sorted(set(room_floor_to_cat.values()))
    letter_to_int = {letter: i for i, letter in enumerate(letters)}

    # 使用映射转换字典的值
    converted_dict = {k: letter_to_int[v] for k, v in room_floor_to_cat.items()}
    room_ids__ = list(converted_dict.keys())
    floor_values = list(converted_dict.values())
    print("room_ids__=",room_ids__)
    print("room_floor_to_cat=",room_floor_to_cat)
    print("#############floor_values=",floor_values)

    # print("room_floor_to_cat=",room_floor_to_cat)
    with h5py.File("data/semantic_maps/gibson/point_clouds/Darden_room.h5", "w") as fp:
        fp.create_dataset("vertices", data=vertices_room)
        fp.create_dataset("obj_ids", data=room_ids)
        fp.create_dataset("sem_ids", data=sem_room_ids)
        fp.create_dataset("colors", data=colors_room)
        fp.create_dataset("room_floor_number", data=floor_values)
        # fp.create_dataset("floor_number",data=floor_number)
        # fp.create




def _aux_fn(inputs):
    return inputs[0](*inputs[1:])


def extract_wall_point_clouds(
    glb_path,
    houses_dim_path,
    sampling_density=1600.0,
    grid_size=2.0,
):
    env = glb_path.split("/")[-1].split(".")[0]

    # Get house dimensions
    houses_dim = json.load(open(houses_dim_path, "r"))
    # Generate floor-wise point-clouds
    per_floor_dims = {}
    for key, val in houses_dim.items():
        match = re.search(f"{env}_(\d+)", key)
        if match:
            per_floor_dims[int(match.group(1))] = val

    # For each floor in the building, get (x, z) specific y-values for nav locations.
    sim = hab_utils.robust_load_sim(glb_path)
    navmesh_triangles = np.array(sim.pathfinder.build_navmesh_vertices())
    navmesh_vertices = hab_utils.dense_sampling_trimesh(
        navmesh_triangles, sampling_density
    )
    sim.close()
    per_floor_xz_map = {}
    nav_points_per_floor = {}
    for floor_id, floor_dims in per_floor_dims.items():
        floor_navmesh_vertices = navmesh_vertices[
            (navmesh_vertices[:, 1] >= floor_dims["ylo"])
            & (navmesh_vertices[:, 1] < floor_dims["yhi"])
        ]
        nav_points_per_floor[floor_id] = floor_navmesh_vertices
        # Divide into 0.5m x 0.5m grid cells
        floor_x = np.rint(floor_navmesh_vertices[:, 0] / grid_size).astype(np.int32)
        floor_z = np.rint(floor_navmesh_vertices[:, 2] / grid_size).astype(np.int32)
        floor_y = floor_navmesh_vertices[:, 1]
        floor_xz_sets = set(zip(floor_x, floor_z))
        floor_xz_map = {}
        for x, z in floor_xz_sets:
            mask = (floor_x == x) & (floor_z == z)
            floor_xz_map[(x, z)] = np.median(floor_y[mask])
        per_floor_xz_map[floor_id] = floor_xz_map

    # Get all mesh triangles in the scene
    scene = trimesh.load(glb_path)
    wall_pc = hab_utils.dense_sampling_trimesh(scene.triangles, sampling_density)
    # Convert coordinate systems
    wall_pc = np.stack([wall_pc[:, 0], wall_pc[:, 2], -wall_pc[:, 1]], axis=1)

    ############################################################################
    # Assign wall points to floors
    ############################################################################
    per_floor_point_clouds = defaultdict(list)
    for floor_id, floor_dims in per_floor_dims.items():
        # Identify points belonging to this floor
        curr_floor_y = floor_dims["ylo"]
        if floor_id + 1 in per_floor_dims:
            next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
        else:
            next_floor_y = math.inf
        floor_mask = (curr_floor_y <= wall_pc[:, 1]) & (
            wall_pc[:, 1] <= next_floor_y - 0.5
        )
        floor_pc = wall_pc[floor_mask, :]
        floor_xz_map = per_floor_xz_map[floor_id]
        # Decide whether each point is a wall point or not
        floor_x_disc = np.around(floor_pc[:, 0] / grid_size).astype(np.int32)
        floor_z_disc = np.around(floor_pc[:, 2] / grid_size).astype(np.int32)
        floor_y = floor_pc[:, 1]
        mask = np.zeros(floor_y.shape[0], dtype=np.bool)
        for i, (x_disc, z_disc, y) in enumerate(
            zip(floor_x_disc, floor_z_disc, floor_y)
        ):
            floor_y = per_floor_dims[floor_id]["ylo"]
            if (x_disc, z_disc) in floor_xz_map:
                floor_y = floor_xz_map[(x_disc, z_disc)]
            # Add point if within height thresholds
            if WALL_THRESH[0] <= y - floor_y < WALL_THRESH[1]:
                mask[i] = True
        per_floor_point_clouds[floor_id] = floor_pc[mask]

    return per_floor_point_clouds


def get_scene_boundaries(inputs):
    scene_path, save_path = inputs
    # print("scene_path,save_path=",scene_path,save_path)
    # scene_path,save_path= data/scene_datasets/gibson_semantic/Beechwood.glb 
    # data/semantic_maps/gibson/scene_boundaries/Beechwood.json
    sim = hab_utils.robust_load_sim(scene_path)
    # print("$$$$$$$$$$$$$ into scene boundary ")
    # print("sim=",sim)
    floor_exts = hab_utils.get_floor_heights(
        sim, sampling_resolution=SAMPLING_RESOLUTION
    )
    # print("floor_exts=",floor_exts)
    scene_name = scene_path.split("/")[-1].split(".")[0]
    # print(scene_name)

    def convert_lu_bound_to_smnet_bound(
        lu_bound, buf=np.array([3.0, 0.0, 3.0])  # meters
    ):
        lower_bound = lu_bound[0] - buf
        upper_bound = lu_bound[1] + buf
        smnet_bound = {
            "xlo": lower_bound[0].item(),
            "ylo": lower_bound[1].item(),
            "zlo": lower_bound[2].item(),
            "xhi": upper_bound[0].item(),
            "yhi": upper_bound[1].item(),
            "zhi": upper_bound[2].item(),
            "center": ((lower_bound + upper_bound) / 2.0).tolist(),
            "sizes": np.abs(upper_bound - lower_bound).tolist(),
        }
        return smnet_bound

    bounds = hab_utils.get_navmesh_extents_at_y(sim, y_bounds=None)
    print("bounds=",bounds)
    # bounds= (array([-17.073181 ,  -2.3777478,  -4.210206 ], dtype=float32), array([3.104614 , 5.8968177, 3.486755 ], dtype=float32))

    scene_boundaries = {}
    scene_boundaries[scene_name] = convert_lu_bound_to_smnet_bound(bounds)
    for fidx, fext in enumerate(floor_exts):
        bounds = hab_utils.get_navmesh_extents_at_y(
            sim, y_bounds=(fext["min"] - 0.5, fext["max"] + 0.5)
        )
        scene_boundaries[f"{scene_name}_{fidx}"] = convert_lu_bound_to_smnet_bound(
            bounds
        )
    
    # scene_boundaries_room = {}
    # # print("scene=",scene_name)
    # scene_name = scene_name+str("_room")
    # # print("scene=",scene_name)
    # scene_boundaries_room[scene_name] = convert_lu_bound_to_smnet_bound(bounds)
    # for fidx, fext in enumerate(floor_exts):
    #     bounds = hab_utils.get_navmesh_extents_at_y(
    #         sim, y_bounds=(fext["min"] - 0.5, fext["max"] + 0.5)
    #     )
    #     scene_boundaries_room[f"{scene_name}_{fidx}"] = convert_lu_bound_to_smnet_bound(
    #         bounds
    #     )

    

    sim.close()
    # print("will save origianl boundary")
    with open(save_path, "w") as fp:
        json.dump(scene_boundaries, fp)
    # print("will save room map boundary")

    # with open("data/semantic_maps/gibson/scene_boundaries/Darden_room.json", "w") as fp:
    #     json.dump(scene_boundaries_room, fp)


def visualize_sem_map(sem_map):
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in COLOR_PALETTE]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    palette_img = get_palette_image()
    H = palette_img.shape[0]
    W = float(palette_img.shape[0]) * semantic_img.shape[1] / semantic_img.shape[0]
    W = int(W)
    semantic_img = cv2.resize(semantic_img, (W, H))
    semantic_img = np.concatenate([semantic_img, palette_img], axis=1)

    return semantic_img



def visualize_sem_map_room(sem_map):
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in COLOR_PALETTE_ROOM]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)
    palette_img = get_palette_image_room()
    H = palette_img.shape[0]
    W = float(palette_img.shape[0]) * semantic_img.shape[1] / semantic_img.shape[0]
    W = int(W)
    semantic_img = cv2.resize(semantic_img, (W, H))
    semantic_img = np.concatenate([semantic_img, palette_img], axis=1)

    return semantic_img


def convert_point_cloud_to_semantic_map(
    pc_dir, houses_dim_root, save_dir, resolution=0.05
):
    # print("pc_dir=",pc_dir)
    # print("house_dim_root=",houses_dim_root)



    # obj_files = sorted(glob.glob(f"{pc_dir}/*.h5"))
    obj_files= ['data/semantic_maps/gibson/point_clouds/Darden.h5']
    # print("obj_file=",obj_files)

    info = {}

    for obj_f in tqdm.tqdm(obj_files):

        env = obj_f.split("/")[-1].split(".")[0]
        # print("env=",env)
        map_save_path = os.path.join(save_dir, env + ".h5")

        # if os.path.isfile(map_save_path):
        #     continue

        with open(os.path.join(houses_dim_root, env + ".json"), "r") as fp:
            houses_dim = json.load(fp)
            # print("houses_dim_root=",houses_dim_root)
            # print("house_dim=",houses_dim)
        # env= Beechwood
        # houses_dim_root= data/semantic_maps/gibson/scene_boundaries
        f = h5py.File(obj_f, "r")

        # Generate floor-wise maps
        per_floor_dims = {}
        for key, val in houses_dim.items():
            match = re.search(f"{env}_(\d+)", key)
            if match:
                per_floor_dims[int(match.group(1))] = val
        # print("per_floor_dims=",per_floor_dims)

        all_vertices = np.array(f["vertices"])
        all_obj_ids = np.array(f["obj_ids"])
        # print("all_obj_ids=",all_obj_ids)
        all_sem_ids = np.array(f["sem_ids"])
        # print("all_sem_ids=",all_sem_ids)
        all_colors = np.array(f["colors"])

        f.close()

        # --- change coordinates to match map
        # --  set discret dimensions
        center = np.array(houses_dim[env]["center"])
        sizes = np.array(houses_dim[env]["sizes"])
        sizes += 2  # -- pad env bboxes

        world_dim = sizes.copy()
        world_dim[1] = 0

        central_pos = center.copy()
        central_pos[1] = 0

        map_world_shift = central_pos - world_dim / 2

        world_dim_discret = [
            int(np.round(world_dim[0] / resolution)),
            0,
            int(np.round(world_dim[2] / resolution)),
        ]

        info[env] = {
            "dim": world_dim_discret,
            "central_pos": [float(x) for x in central_pos],
            "map_world_shift": [float(x) for x in map_world_shift],
            "resolution": resolution,
        }
        print("info=",info[env])

        # Pre-assign objects to different floors
        per_floor_obj_ids = {floor_id: [] for floor_id in per_floor_dims.keys()}
        # print("per_floor_obj_ids=",per_floor_obj_ids)
        obj_ids_set = set(all_obj_ids.tolist())
        # print("obj_ids_set=",obj_ids_set)
        ## -1 corresponds to wall and floor
        if -1 in obj_ids_set:
            obj_ids_set.remove(-1)
        for obj_id in obj_ids_set:
            # print("obj")
            is_obj_id = all_obj_ids == obj_id
            # print("is_obj_id=",is_obj_id)
            # print("obj_id=",obj_id)
            # print("all_obj_ids=",all_obj_ids)

            obj_vertices = all_vertices[is_obj_id, :]
            # print("obj_vertices=",obj_vertices.shape)


            # Get extents
            min_y = obj_vertices[:, 1].min()
            # Assign object to floor closest to it's min_y
            best_floor_id = None
            best_diff = math.inf
            # print("per_floor_dims=",per_floor_dims)
            for floor_id, floor_dims in per_floor_dims.items():
                diff = abs(min_y - floor_dims["ylo"])
                # print("min_y=",min_y)
                # print("floor_dims[ylo]=",floor_dims["ylo"])
                # print("diff=",diff)
                if (diff < best_diff) and min_y - floor_dims["ylo"] > -0.5:
                    best_diff = diff
                    best_floor_id = floor_id
                    # print("best floor id =",best_floor_id)
            if best_floor_id is None:
                # Skip the object if it does not belong to any floor
                # Print message for debugging purposes
                print(
                    f"NOTE: Object id {obj_id} from scene {env} does not belong to any floor!"
                )
                continue
            # print("best_floor_id=",best_floor_id)
            # print("obj_id=",obj_id)
            per_floor_obj_ids[best_floor_id].append(obj_id)
            # print("per_floor_obj_ids_FOR_OBJ=",per_floor_obj_ids)

        # Build maps per floor
        per_floor_maps = {}
        for floor_id, floor_dims in per_floor_dims.items():
            # print("floor_id=",floor_id)
            # print("floor_dims=",floor_dims)

            curr_floor_y = floor_dims["ylo"]
            if floor_id + 1 in per_floor_dims:
                next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y = math.inf
            # Get navigable and wall vertices based on height thresholds
            is_on_floor = (all_vertices[:, 1] >= curr_floor_y) & (
                all_vertices[:, 1] <= next_floor_y - 0.5
            )
            is_floor = (all_sem_ids == OBJECT_CATEGORY_MAP["floor"]) & is_on_floor
            is_wall = (all_sem_ids == OBJECT_CATEGORY_MAP["wall"]) & is_on_floor

            # Get object vertices based on height thresholds for individual object instances
            is_object = np.zeros_like(is_on_floor)
            for obj_id in per_floor_obj_ids[floor_id]:
                is_object = is_object | (all_obj_ids == obj_id)

            mask = is_floor | is_wall | is_object

            vertices = np.copy(all_vertices[mask])
            obj_ids = np.copy(all_obj_ids[mask])
            sem_ids = np.copy(all_sem_ids[mask])

            # -- some maps have 0 obj of interest
            if len(vertices) == 0:
                info[env][floor_id] = {"y_min": 0.0}
                dims = (world_dim_discret[2], world_dim_discret[0])
                mask = np.zeros(dims, dtype=bool)
                map_z = np.zeros(dims, dtype=np.float32)
                map_instance = np.zeros(dims, dtype=np.int32)
                map_semantic = np.zeros(dims, dtype=np.int32)
                map_semantic_rgb = np.zeros((*dims, 3), dtype=np.uint8)
                per_floor_maps[floor_id] = {
                    "mask": mask,
                    "map_z": map_z,
                    "map_instance": map_instance,
                    "map_semantic": map_semantic,
                    "map_semantic_rgb": map_semantic_rgb,
                }
                continue

            vertices -= map_world_shift

            # Set the min_y for the floor. This will be used during episode generation to find
            # a random navigable start location.
            floor_mask = sem_ids == OBJECT_CATEGORY_MAP["floor"]
            # print("floor_mask=",floor_mask)
            min_y = vertices[floor_mask, 1].min()
            # print("min_y=",min_y)
            info[env][floor_id] = {"y_min": float(min_y.item())}
            # print("info[env][floor_id]=",info[env][floor_id])

            # Reduce heights of floor and navigable space to ensure objects are taller.
            wall_mask = sem_ids == OBJECT_CATEGORY_MAP["wall"]
            vertices[wall_mask, 1] -= 0.5
            vertices[floor_mask, 1] -= 0.5

            # -- discretize point cloud
            vertices = torch.FloatTensor(vertices)
            obj_ids = torch.FloatTensor(obj_ids)
            sem_ids = torch.FloatTensor(sem_ids)

            y_values = vertices[:, 1]

            vertex_to_map_x = (vertices[:, 0] / resolution).round()
            vertex_to_map_z = (vertices[:, 2] / resolution).round()

            outside_map_indices = (
                (vertex_to_map_x >= world_dim_discret[0])
                + (vertex_to_map_z >= world_dim_discret[2])
                + (vertex_to_map_x < 0)
                + (vertex_to_map_z < 0)
            )

            # assert outside_map_indices.sum() == 0
            y_values = y_values[~outside_map_indices]
            vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
            vertex_to_map_x = vertex_to_map_x[~outside_map_indices]

            obj_ids = obj_ids[~outside_map_indices]
            sem_ids = sem_ids[~outside_map_indices]

            # -- get the z values for projection
            # -- shift to positive values
            y_values = y_values - min_y
            y_values += 1.0

            # -- projection
            feat_index = (
                world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x
            ).long()
            flat_highest_z = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_highest_z, argmax_flat_spatial_map = scatter_max(
                y_values,
                feat_index,
                dim=0,
                out=flat_highest_z,
            )
            # NOTE: This is needed only for torch_scatter>=2.3
            argmax_flat_spatial_map[argmax_flat_spatial_map == y_values.shape[0]] = -1

            m = argmax_flat_spatial_map >= 0
            flat_map_instance = (
                torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
            )

            flat_map_instance[m.view(-1)] = obj_ids[argmax_flat_spatial_map[m]]

            flat_map_semantic = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_map_semantic[m.view(-1)] = sem_ids[argmax_flat_spatial_map[m]]

            # -- format data
            mask = m.reshape(world_dim_discret[2], world_dim_discret[0])
            mask = mask.numpy()
            mask = mask.astype(bool)
            map_z = flat_highest_z.reshape(world_dim_discret[2], world_dim_discret[0])
            map_z = map_z.numpy()
            map_z = map_z.astype(np.float32)
            map_instance = flat_map_instance.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_instance = map_instance.numpy()
            map_instance = map_instance.astype(np.float32)
            map_semantic = flat_map_semantic.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_semantic = map_semantic.numpy()
            map_semantic = map_semantic.astype(np.float32)
            map_semantic_rgb = visualize_sem_map(map_semantic)

            per_floor_maps[floor_id] = {
                "mask": mask,
                "map_z": map_z,
                "map_instance": map_instance,
                "map_semantic": map_semantic,
                "map_semantic_rgb": map_semantic_rgb,
            }

            rgb_save_path = os.path.join(save_dir, f"{env}_{floor_id}.png")
            cv2.imwrite(rgb_save_path, map_semantic_rgb)

        with h5py.File(map_save_path, "w") as f:
            # print("map_save_path=",map_save_path)
            f.create_dataset(f"wall_sem_id", data=OBJECT_CATEGORY_MAP["wall"])
            f.create_dataset(f"floor_sem_id", data=OBJECT_CATEGORY_MAP["floor"])
            f.create_dataset(
                f"out-of-bounds_sem_id", data=OBJECT_CATEGORY_MAP["out-of-bounds"]
            )
            for floor_id, floor_map in per_floor_maps.items():
                mask = floor_map["mask"]
                map_z = floor_map["map_z"]
                map_instance = floor_map["map_instance"]
                map_semantic = floor_map["map_semantic"]
                map_semantic_rgb = floor_map["map_semantic_rgb"]

                f.create_dataset(f"{floor_id}/mask", data=mask, dtype=bool)
                f.create_dataset(
                    f"{floor_id}/map_heights", data=map_z, dtype=np.float32
                )
                f.create_dataset(
                    f"{floor_id}/map_instance", data=map_instance, dtype=np.int32
                )
                f.create_dataset(
                    f"{floor_id}/map_semantic", data=map_semantic, dtype=np.int32
                )
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=map_semantic_rgb)

    json.dump(info, open(os.path.join(save_dir, "semmap_GT_info.json"), "w"))



def convert_point_cloud_to_semantic_room_map(
    pc_dir, houses_dim_root, save_dir, resolution=0.05
):
    # print("pc_dir=",pc_dir)
    # print("house_dim_root=",houses_dim_root)


    # global room_floor_cat

    # print("@@@@@@@@@@@@@@@@@@@room_floor_cat=",room_floor_cat)
    # obj_files = sorted(glob.glob(f"{pc_dir}/*_room.h5"))
    obj_files= ['data/semantic_maps/gibson/point_clouds/Darden_room.h5']    
    # print("obj_file=",obj_files)

    info = {}

    for obj_f in tqdm.tqdm(obj_files):
        print("obj_f=",obj_f)
      

        env = obj_f.split("/")[-1].split(".")[0]

        print("env1=",env)
        
        map_save_path = os.path.join(save_dir, env + ".h5")
        if os.path.isfile(map_save_path):
            continue

        
        # with open(os.path.join(houses_dim_root, env + ".json"), "r") as fp:
        with open(os.path.join("data/semantic_maps/gibson/scene_boundaries/Darden.json"), "r") as fp:

            # data/semantic_maps/gibson/scene_boundaries/Darden_room.json
            # print("fp=",fp)
            houses_dim = json.load(fp)
        f = h5py.File(obj_f, "r")
        env = env.replace("_room","")
        print("env2=",env)

        # Generate floor-wise maps
        per_floor_dims = {}
        for key, val in houses_dim.items():
            match = re.search(f"{env}_(\d+)", key)
            if match:
                per_floor_dims[int(match.group(1))] = val

        floor_number_cat = np.array(f["room_floor_number"])
        # print("@@@@@@@@@@@@@@@@@@@@floor_number=",floor_number_cat)

        all_vertices = np.array(f["vertices"])
        # print("all_vertives =",all_vertices.shape)
        all_obj_ids = np.array(f["obj_ids"])
        # print("all_obj_ids=",all_obj_ids.shape)
        all_sem_ids = np.array(f["sem_ids"])
        # print("all_sem_ids=",all_sem_ids.shape)
        all_colors = np.array(f["colors"])

        f.close()

        # env = env +str("_room")
        print("env3=",env)

        # --- change coordinates to match map
        # --  set discret dimensions
        center = np.array(houses_dim[env]["center"])
        sizes = np.array(houses_dim[env]["sizes"])
        sizes += 2  # -- pad env bboxes

        world_dim = sizes.copy()
        world_dim[1] = 0

        central_pos = center.copy()
        central_pos[1] = 0

        map_world_shift = central_pos - world_dim / 2

        world_dim_discret = [
            int(np.round(world_dim[0] / resolution)),
            0,
            int(np.round(world_dim[2] / resolution)),
        ]
        env = env +str("_room")

        info[env] = {
            "dim": world_dim_discret,
            "central_pos": [float(x) for x in central_pos],
            "map_world_shift": [float(x) for x in map_world_shift],
            "resolution": resolution,
        }
        print("info_roommap=",info[env])
        # Pre-assign objects to different floors
        per_floor_obj_ids = {floor_id: [] for floor_id in per_floor_dims.keys()}

# need to do automatically!@!!!!!
        # per_floor_obj_ids= {0: []}
        

        # print("per_floor_obj_ids=",per_floor_obj_ids)
        # print("per_floor_dims=",per_floor_dims)
        


        # per_floor_dims= {0: {'xlo': -13.815871238708496, 'ylo': 0.014301702380180359, 'zlo': -8.274190902709961, 'xhi': 4.134128570556641, 'yhi': 1.2143017053604126, 'zhi': 9.525810241699219, 'center': [-4.840871334075928, 0.6143017038702965, 0.6258096694946289], 'sizes': [17.949999809265137, 1.2000000029802322, 17.80000114440918]}, 1: {'xlo': -13.815871238708496, 'ylo': 2.6143016815185547, 'zlo': -12.574191093444824, 'xhi': 5.2841291427612305, 'yhi': 3.614301919937134, 'zhi': 9.525810241699219, 'center': [-4.265871047973633, 3.1143018007278442, -1.5241904258728027], 'sizes': [19.100000381469727, 1.000000238418579, 22.100001335144043]}}
        
        
        obj_ids_set = set(all_obj_ids.tolist())
        ## -1 corresponds to wall and floor
        if -1 in obj_ids_set:
            obj_ids_set.remove(-1)
        for obj_id in obj_ids_set:
            
            is_obj_id = all_obj_ids == obj_id
            # print("obj_id=",obj_id)
            # print("all_obj_ids=",all_obj_ids.shape)
            # print("is_obj_id=",is_obj_id)
            # print()



            obj_vertices = all_vertices[is_obj_id, :]
            # print("obj_vetices=",obj_vertices.shape)
            # Get extents
            min_y = obj_vertices[:, 1].min()
            # print("min_y=",min_y)
            # Assign object to floor closest to it's min_y
            best_floor_id = None
            best_diff = math.inf
            # print("best_diff=",best_diff)
            # print("##############11")
            # for floor_id, floor_dims in per_floor_dims.items():
            #     print("##############")
            #     print("floor_id=",floor_id)
            #     diff = abs(min_y - floor_dims["ylo"])
            #     print("min_y=",min_y)
            #     print("floor_dim ylow=",floor_dims["ylo"])
            #     print("diff=",diff)
            #     if (diff < best_diff) and min_y - floor_dims["ylo"] > -0.5:
            #         best_diff = diff
            #         best_floor_id = floor_id
            #         print("best diff=",best_diff)
            #         print("floor_id=",floor_id)
            #         print("best floor id=",best_floor_id)
            # # if best_floor_id is None:
            # #     # Skip the object if it does not belong to any floor
            # #     # Print message for debugging purposes
            # #     print(
            # #         f"NOTE: Object id {obj_id} from scene {env} does not belong to any floor!"
            # #     )
            # #     continue
            best_floor_id = floor_number_cat[obj_id-1]
            # print("floor_number shape =",floor_number.shape)
            # print("best floor id=",best_floor_id)
            # print("obj_id=",obj_id)

            ##### read form npz file to get floor number!!!!!


            per_floor_obj_ids[best_floor_id].append(obj_id)
            # print("per_floor_obj_ids=",per_floor_obj_ids)
            # per_floor_obj_ids= {0: [None], 1: [None]}

        # Build maps per floor
        per_floor_maps = {}
        for floor_id, floor_dims in per_floor_dims.items():
            # print("floor_id=",floor_id)
            # print("floor_dims=",floor_dims)

            curr_floor_y = floor_dims["ylo"]
            # print("curr_floor_y=",curr_floor_y)

            if floor_id + 1 in per_floor_dims:
                next_floor_y = per_floor_dims[floor_id + 1]["ylo"]
            else:
                next_floor_y = math.inf
            # print("next_floor_y=",next_floor_y)

            # Get navigable and wall vertices based on height thresholds
            is_on_floor = (all_vertices[:, 1] >= curr_floor_y) & (
                all_vertices[:, 1] <= next_floor_y -0.5
                # all_vertices[:, 1] <= next_floor_y +2

            )
            # print("all_vertices[:, 1]=",all_vertices[:, 1])

            is_floor = (all_sem_ids == OBJECT_CATEGORY_MAP["floor"]) & is_on_floor
            # print("is floor=",is_floor,is_floor.shape)

            is_wall = (all_sem_ids == GIBSON_ROOM_CATEGORY_MAP["wall"]) & is_on_floor
            # print("is wall=",is_wall,is_wall.shape)
            # Get object vertices based on height thresholds for individual object instances
            is_object = np.zeros_like(is_on_floor)
            # print("is object=",is_object,is_object.shape)
            for obj_id in per_floor_obj_ids[floor_id]:
                # print("obj_id=",obj_id)
                # print(" per_floor_obj_ids[floor_id]=", per_floor_obj_ids[floor_id])
                is_object = is_object | (all_obj_ids == obj_id)
                # print("all_obj_ids=",all_obj_ids,all_obj_ids.shape)
                # print("is_objext=",is_object)

            mask = is_floor | is_wall | is_object
          
            # mask =  is_wall | is_object
            # mask =  is_object

            # print("is_floor=",is_floor)
            # print("is_wall=",is_wall)
            # print("is_object=",is_object)
            # print("mask=",mask)


            # mask = is_wall | is_object


            vertices = np.copy(all_vertices[mask])
            obj_ids = np.copy(all_obj_ids[mask])
            sem_ids = np.copy(all_sem_ids[mask])

            # -- some maps have 0 obj of interest
            if len(vertices) == 0:
                info[env][floor_id] = {"y_min": 0.0}
                dims = (world_dim_discret[2], world_dim_discret[0])
                mask = np.zeros(dims, dtype=bool)
                map_z = np.zeros(dims, dtype=np.float32)
                map_instance = np.zeros(dims, dtype=np.int32)
                map_semantic = np.zeros(dims, dtype=np.int32)
                map_semantic_rgb = np.zeros((*dims, 3), dtype=np.uint8)
                per_floor_maps[floor_id] = {
                    "mask": mask,
                    "map_z": map_z,
                    "map_instance": map_instance,
                    "map_semantic": map_semantic,
                    "map_semantic_rgb": map_semantic_rgb,
                }
                continue

            vertices -= map_world_shift

            # Set the min_y for the floor. This will be used during episode generation to find
            # a random navigable start location.
            floor_mask = sem_ids == GIBSON_ROOM_CATEGORY_MAP["floor"]
            # print("floor_mask sem_id=",sem_ids)
            min_y = vertices[floor_mask, 1].min()
            info[env][floor_id] = {"y_min": float(min_y.item())}

            # Reduce heights of floor and navigable space to ensure objects are taller.
            wall_mask = sem_ids == GIBSON_ROOM_CATEGORY_MAP["wall"]
            # print("wall mask sem_ids =",sem_ids)
            vertices[wall_mask, 1] -= -1
            # vertices[wall_mask, 1] -= 0.5

            vertices[floor_mask, 1] -= 0.5

            # -- discretize point cloud
            vertices = torch.FloatTensor(vertices)
            obj_ids = torch.FloatTensor(obj_ids)
            sem_ids = torch.FloatTensor(sem_ids)

            y_values = vertices[:, 1]

            vertex_to_map_x = (vertices[:, 0] / resolution).round()
            vertex_to_map_z = (vertices[:, 2] / resolution).round()

            outside_map_indices = (
                (vertex_to_map_x >= world_dim_discret[0])
                + (vertex_to_map_z >= world_dim_discret[2])
                + (vertex_to_map_x < 0)
                + (vertex_to_map_z < 0)
            )

            # assert outside_map_indices.sum() == 0
            y_values = y_values[~outside_map_indices]
            vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
            vertex_to_map_x = vertex_to_map_x[~outside_map_indices]

            obj_ids = obj_ids[~outside_map_indices]
            sem_ids = sem_ids[~outside_map_indices]

            # -- get the z values for projection
            # -- shift to positive values
            y_values = y_values - min_y
            y_values += 1.0

            # -- projection
            feat_index = (
                world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x
            ).long()
            flat_highest_z = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_highest_z, argmax_flat_spatial_map = scatter_max(
                y_values,
                feat_index,
                dim=0,
                out=flat_highest_z,
            )
            # NOTE: This is needed only for torch_scatter>=2.3
            argmax_flat_spatial_map[argmax_flat_spatial_map == y_values.shape[0]] = -1

            m = argmax_flat_spatial_map >= 0
            flat_map_instance = (
                torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
            )

            flat_map_instance[m.view(-1)] = obj_ids[argmax_flat_spatial_map[m]]

            flat_map_semantic = torch.zeros(
                int(world_dim_discret[0] * world_dim_discret[2])
            )
            flat_map_semantic[m.view(-1)] = sem_ids[argmax_flat_spatial_map[m]]

            # -- format data
            mask = m.reshape(world_dim_discret[2], world_dim_discret[0])
            mask = mask.numpy()
            mask = mask.astype(bool)
            map_z = flat_highest_z.reshape(world_dim_discret[2], world_dim_discret[0])
            map_z = map_z.numpy()
            map_z = map_z.astype(np.float32)
            map_instance = flat_map_instance.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_instance = map_instance.numpy()
            map_instance = map_instance.astype(np.float32)
            map_semantic = flat_map_semantic.reshape(
                world_dim_discret[2], world_dim_discret[0]
            )
            map_semantic = map_semantic.numpy()
            map_semantic = map_semantic.astype(np.float32)
            map_semantic_rgb = visualize_sem_map_room(map_semantic)

            per_floor_maps[floor_id] = {
                "mask": mask,
                "map_z": map_z,
                "map_instance": map_instance,
                "map_semantic": map_semantic,
                "map_semantic_rgb": map_semantic_rgb,
            }

            rgb_save_path = os.path.join(save_dir, f"{env}_{floor_id}.png")
            cv2.imwrite(rgb_save_path, map_semantic_rgb)

        map_save_path_room= "data/semantic_maps/gibson/semantic_maps/Darden_room.h5"
        with h5py.File(map_save_path_room, "w") as f:
            f.create_dataset(f"wall_sem_id", data=GIBSON_ROOM_CATEGORY_MAP["wall"])
            f.create_dataset(f"floor_sem_id", data=GIBSON_ROOM_CATEGORY_MAP["floor"])
            f.create_dataset(
                f"out-of-bounds_sem_id", data=GIBSON_ROOM_CATEGORY_MAP["out-of-bounds"]
            )
            for floor_id, floor_map in per_floor_maps.items():
                mask = floor_map["mask"]
                map_z = floor_map["map_z"]
                map_instance = floor_map["map_instance"]
                map_semantic = floor_map["map_semantic"]
                map_semantic_rgb = floor_map["map_semantic_rgb"]

                f.create_dataset(f"{floor_id}/mask", data=mask, dtype=bool)
                f.create_dataset(
                    f"{floor_id}/map_heights", data=map_z, dtype=np.float32
                )
                f.create_dataset(
                    f"{floor_id}/map_instance", data=map_instance, dtype=np.int32
                )
                f.create_dataset(
                    f"{floor_id}/map_semantic", data=map_semantic, dtype=np.int32
                )
                f.create_dataset(f"{floor_id}/map_semantic_rgb", data=map_semantic_rgb)

    json.dump(info, open(os.path.join(save_dir, "semmap_room_GT_info.json"), "w"))








if __name__ == "__main__":
    scene_paths = sorted(
        glob.glob(
            os.path.join(SCENES_ROOT, "**/*.glb"),
            recursive=True,
        )
    )
    # print("scene_path=",scene_paths)
    # Select only scenes that have corresponding semantics
    scene_paths = list(
        filter(
            lambda x: os.path.isfile(x.replace(".glb", "_semantic.ply")), scene_paths
        )
    )
    # print("scene_path2=",scene_paths)
    # Select only scenes from the train and val splits
    valid_scenes = (
        SPLIT_SCENES[ACTIVE_DATASET]["train"] + SPLIT_SCENES[ACTIVE_DATASET]["val"]
    )
    scene_paths = list(
        filter(lambda x: os.path.basename(x).split(".")[0] in valid_scenes, scene_paths)
    )

    print(f"Number of available scenes: {len(scene_paths)}")

    context = mp.get_context("forkserver")
    # print("context=",context)
    pool = context.Pool(NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD)
    # print("pool=",pool)

    # Extract scene_boundaries
    os.makedirs(SB_SAVE_ROOT, exist_ok=True)
    print("===========> Extracting scene boundaries")
    inputs = []
    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path).split(".")[0]
        save_path = os.path.join(SB_SAVE_ROOT, f"{scene_name}.json")
        if not os.path.isfile(save_path):
            inputs.append((scene_path, save_path))
    _ = list(tqdm.tqdm(pool.imap(get_scene_boundaries, inputs), total=len(inputs)))
    print("_ =",_)

    # Generate point-clouds for each scene
    os.makedirs(PC_SAVE_ROOT, exist_ok=True)
    print("===========> Extracting point-clouds")
    inputs = []
    for scene_path in scene_paths:
        ply_path = scene_path.replace(".glb", "_semantic.ply")
        # print("ply_path=",ply_path)
        scn_path = scene_path.replace(".glb", ".scn")
        # print("scn_path=",scn_path)
        scene_name = scene_path.split("/")[-1].split(".")[0]
        pc_save_path = os.path.join(PC_SAVE_ROOT, f"{scene_name}.h5")
        if not os.path.isfile(pc_save_path):
            inputs.append(
                (
                    extract_scene_point_clouds,
                    scene_path,
                    ply_path,
                    scn_path,
                    os.path.join(SB_SAVE_ROOT, f"{scene_name}.json"),
                    pc_save_path,
                )
            )

    _ = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(inputs)))

    # Extract semantic maps
    os.makedirs(SEM_SAVE_ROOT, exist_ok=True)
    print("===========> Extracting semantic maps")
    print("PC_SAVE_ROOT=",PC_SAVE_ROOT)
    print("SB_SAVE_ROOT=",SB_SAVE_ROOT)
    print("SEM_SAVE_ROOT=",SEM_SAVE_ROOT)
    convert_point_cloud_to_semantic_map(PC_SAVE_ROOT, SB_SAVE_ROOT, SEM_SAVE_ROOT)
    convert_point_cloud_to_semantic_room_map(PC_SAVE_ROOT, SB_SAVE_ROOT, SEM_SAVE_ROOT)
