# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import numpy as np 
from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc







# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {

    "coco_2017_train": ("/home/anishmad/msr_thesis/glip/DATASET/coco/train2017", "/home/anishmad/msr_thesis/glip/DATASET/coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("/home/anishmad/msr_thesis/glip/DATASET/coco/val2017", "/home/anishmad/msr_thesis/glip/DATASET/coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("/home/anishmad/msr_thesis/glip/DATASET/coco/test2017", "/home/anishmad/msr_thesis/glip/DATASET/coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("/home/anishmad/msr_thesis/glip/DATASET/coco/test2017", "/home/anishmad/msr_thesis/glip/DATASET/coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("/home/anishmad/msr_thesis/glip/DATASET/coco/val2017", "/home/anishmad/msr_thesis/glip/DATASET/coco/annotations/instances_val2017_100.json"),
}
FLIR_CAMERA_OBJECTS_ROOT = '/data3/anishmad/roboflow_data/flir_camera_objects/'

_PREDEFINED_SPLITS_FLIR_CAMERA_OBJECTS = {
    "flir_camera_objects_all_cls_train": (f"{FLIR_CAMERA_OBJECTS_ROOT}/train/", f"{FLIR_CAMERA_OBJECTS_ROOT}/train/_annotations.coco.json"),
    "flir_camera_objects_all_cls_val": (f"{FLIR_CAMERA_OBJECTS_ROOT}/val/", f"{FLIR_CAMERA_OBJECTS_ROOT}/val/_annotations.coco.json"),
    "flir_camera_objects_all_cls_test": (f"{FLIR_CAMERA_OBJECTS_ROOT}/test/", f"{FLIR_CAMERA_OBJECTS_ROOT}/test/_annotations.coco.json"),
}


LIVER_DISEASE_ROOT = '/data3/anishmad/roboflow_data/liver_disease'

_PREDEFINED_SPLITS_LIVER_DISEASE = {
    "liver_all_cls_train": (f"{LIVER_DISEASE_ROOT}/train/", f"{LIVER_DISEASE_ROOT}/train/_annotations.coco.json"),
    "liver_all_cls_val": (f"{LIVER_DISEASE_ROOT}/val/", f"{LIVER_DISEASE_ROOT}/val/_annotations.coco.json"),
    "liver_all_cls_test": (f"{LIVER_DISEASE_ROOT}/test/", f"{LIVER_DISEASE_ROOT}/test/_annotations.coco.json"),
}


NUIMAGES_ROOT = '/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages'
NUIMAGES_ANN_ROOT_NO_WC = '/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/no_wc'


_PREDEFINED_SPLITS_NUIMAGES = {}
_PREDEFINED_SPLITS_NUIMAGES["nuimages_all_cls"] = {
    
    "nuimages_all_cls_train": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ROOT}/annotations/nuimages_v1.0-train.json"),
    "nuimages_all_cls_val": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ROOT}/annotations/nuimages_v1.0-val.json"),
    "nuimages_all_cls_dummy": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ROOT}/annotations/nuimages_dummy_v1.0-val.json"),
    "nuimages_all_cls_dummy_train": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ROOT}/annotations/nuimages_dummy_v1.0-train.json"),
}

_PREDEFINED_SPLITS_NUIMAGES["nuimages_all_cls_no_wc"] = {
    
    "nuimages_all_cls_train_no_wc": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ANN_ROOT_NO_WC}/nuimages_v1.0-train.json"),
    "nuimages_all_cls_val_no_wc": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ANN_ROOT_NO_WC}/nuimages_v1.0-val.json"),
    "nuimages_all_cls_val_no_wc_dummy": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ANN_ROOT_NO_WC}/nuimages_dummy_v1.0-val.json"),
    # "nuimages_all_cls_val_no_wc_challenge": (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/datasets/nuimages/annotations/ffsod_challenge/test_set/test_set.json"),
    "nuimages_all_cls_val_no_wc_challenge": (f"/data3/anishmad/data/nuimages_challenge/test_set_images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages_challenge/test_set_challenge_new.json"),
    "nuimages_all_cls_train_10_shots_no_wc_challenge": (f"/data3/anishmad/data/nuimages_challenge/training_split_10_shots/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages_challenge/10_shots/nuimages_fsod_train_shots_10.json"),
    "nuimages_all_cls_val_10_shots_no_wc_challenge": (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages_support/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_val_shots_10.json"),
    "nuimages_all_cls_val_no_wc_challenge_dummy": (f"{NUIMAGES_ROOT}/images/", f"{NUIMAGES_ANN_ROOT_NO_WC}/nuimages_dummy_v1.0-val.json"),
}

fsod_shots = [5,10,30]
fsod_seeds = np.arange(10)
_PREDEFINED_SPLITS_NUIMAGES_FSOD = {}       # few shot and few image

def get_fsod_split_dict(shots, seeds):
    dd = {}
    for seed in seeds:
        for shot in shots:
            # for mode in ['train', 'val']:
            for mode in ['train']:
                dd_key = f'nuimages_fsod_{mode}_seed_{seed}_shots_{shot}'
                value = (f"{NUIMAGES_ROOT}/images/", f"/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages_support/fsod_data_10_seeds_detectron_noroot/no_wc/nuimages_fsod_{mode}_seed_{seed}_shots_{shot}.json")

                dd[dd_key] = value
    return dd

# few image split (oracle - all annotations for few-images)
def get_fiod_split_dict(shots, seeds):  
    dd = {}
    for seed in seeds:
        for shot in shots:
            # for mode in ['train', 'val']:
            for mode in ['train']:
                dd_key = f'nuimages_fiod_{mode}_seed_{seed}_shots_{shot}'
                value = (f"{NUIMAGES_ROOT}/images/", f"/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages_support/fiod_data_10_seeds_detectron/no_wc/nuimages_fiod_{mode}_seed_{seed}_shots_{shot}.json")

                dd[dd_key] = value
    return dd

_PREDEFINED_SPLITS_NUIMAGES_FSOD["nuimages_fsod"] = get_fsod_split_dict(fsod_shots, fsod_seeds)   #using same shots and seeds as FSOD
_PREDEFINED_SPLITS_NUIMAGES_FSOD["nuimages_fiod"] = get_fiod_split_dict(fsod_shots, fsod_seeds)   #using same shots and seeds as FSOD

_PREDEFINED_SPLITS_NUIMAGES_FSOD["nuimages_fsod_best_split"] = {
    'nuimages_fsod_train_best_split_shots_5': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_train_shots_5.json"),
    # 'nuimages_fsod_val_best_split_shots_5': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_val_shots_5.json"),
    'nuimages_fsod_train_best_split_shots_10': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_train_shots_10.json"),
    # 'nuimages_fsod_val_best_split_shots_10': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_val_shots_10.json"),
    'nuimages_fsod_train_best_split_shots_30': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_train_shots_30.json"),
    # 'nuimages_fsod_val_best_split_shots_30': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fsod_data_10_seeds_detectron_best_split/no_wc/nuimages_fsod_val_shots_30.json")
}

_PREDEFINED_SPLITS_NUIMAGES_FSOD["nuimages_fiod_best_split"] = {
    'nuimages_fiod_train_best_split_shots_5': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_train_shots_5.json"),
    # 'nuimages_fiod_val_best_split_shots_5': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_val_shots_5.json"),
    'nuimages_fiod_train_best_split_shots_10': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_train_shots_10.json"),
    # 'nuimages_fiod_val_best_split_shots_10': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_val_shots_10.json"),
    'nuimages_fiod_train_best_split_shots_30': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_train_shots_30.json"),
    # 'nuimages_fiod_val_best_split_shots_30': (f"{NUIMAGES_ROOT}/images/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/nuimages/fiod_data_10_seeds_detectron_best_split/no_wc/nuimages_fiod_val_shots_30.json")
}


_PREDEFINED_SPLITS_NUSCENES = {}
_PREDEFINED_SPLITS_NUSCENES["nuscenes_all_cls"] = {
    
    "nuscenes_all_cls_train": ("/home/anishmad/msr_thesis/glip/DATASET/nuscenes/images", "/home/anishmad/msr_thesis/glip/DATASET/nuscenes/annotations/nuscenes_infos_train_mono3d.coco.json"),
    "nuscenes_all_cls_val": ("/home/anishmad/msr_thesis/glip/DATASET/nuscenes/images", "/home/anishmad/msr_thesis/glip/DATASET/nuscenes/annotations/nuscenes_infos_val_mono3d.coco.json"),
    "nuscenes_all_cls_dummy": ("/home/anishmad/msr_thesis/glip/DATASET/nuscenes/images", "/home/anishmad/msr_thesis/glip/DATASET/nuscenes/annotations/nuscenes_infos_dummy_val_mono3d.coco.json"),
}



def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_nuimages(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUIMAGES.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )

def register_all_liver_disease(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LIVER_DISEASE.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )
def register_all_flir_camera_objects(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_FLIR_CAMERA_OBJECTS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )

def register_all_nuimages_fsod(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUIMAGES_FSOD.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )

def register_all_nuscenes(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_NUSCENES.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                json_file,   # using absolute paths
                image_root,   # using absolute paths
                offset_in_category=0,           # due to modified annotations
            )

# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
        "lvis_v1_dummy_train": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/lvis_v1_dummy_train.json"),
        "lvis_v1_dummy_val": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/lvis_v1_dummy_val.json"),

    },

    "lvis_v05":{
        "lvis_v05_train": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/lvis_v0.5_train.json"),
        "lvis_v05_val": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/lvis_v0.5_val.json"),
        "lvis_v05_train_norare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/lvis_v0.5_train_norare.json"),
    },

    "lvis_v05_fsod":{
        "lvis_v05_train_fsod": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/my_data/fsod/lvis_train_shots_all_cats.json"),
        "lvis_v05_val_fsod": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/my_data/fsod/lvis_train_shots_all_cats.json"),
        "lvis_v05_test_fsod": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/lvis_v0.5_val.json"),
    },

    "lvis_v05_fsod_same_split":{
        "lvis_v05_train_fsod_ssplit": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/downloaded/lvis_shots.json"),
        "lvis_v05_val_fsod_ssplit": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/downloaded/train_val_split/lvis_val_shots_all_cats.json"),
        "lvis_v05_test_fsod_ssplit": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis_v05/lvis_v0.5_val.json"),
    },
    
    "lvis_v1_rare": {
        "lvis_v1_train_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/lvis_v1_train_novel.json"),
        "lvis_v1_val_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/lvis_v1_val_novel.json"),
        "lvis_v1_test_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/lvis_v1_test_novel.json"),

    },

    # "lvis_v1_fsod_rare": {
    #     "lvis_v1_train_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod/lvis_train_novel_shots.json"),
    #     "lvis_v1_val_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod/lvis_val_novel_shots.json"),
    #     "lvis_v1_test_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/lvis_v1_test_novel.json"), # same as lvis_v1_test_rare

    # },

    ### This is the corrected one: earlier version had train and val same by error.
    "lvis_v1_fsod_rare": {
        "lvis_v1_train_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod2/lvis_train_novel_shots.json"),
        "lvis_v1_val_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod2/lvis_val_novel_shots.json"),
        "lvis_v1_test_fsod_rare": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/lvis_v1_test_novel.json"), # same as lvis_v1_test_rare

    },

    "lvis_v1_fsod_novel_all": {
        "lvis_v1_train_fsod_novel_all": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod/lvis_train_novel_shots_all_cats.json"),
        "lvis_v1_val_fsod_novel_all": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/novel_data/fsod/lvis_val_novel_shots_all_cats.json"),
        "lvis_v1_test_fsod_novel_all": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/my_data/lvis_v1_test_novel_all_cats.json"), # same as lvis_v1_test_rare

    },

    # "lvis_v1_fsod": {
    #     "lvis_v1_fsod_train_novel": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/lvis_train_novel_shots.json"),
    #     "lvis_v1_fsod_val_novel": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/lvis_val_novel_shots.json"),
    #     "lvis_v1_fsod_test": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/lvis_v1_test_novel.json"),
       
    # },
    # "lvis_v1_rare":{
    #     "lvis_v1_train_novel_rare_cats": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/only_rare/lvis_train_novel_shots.json"),
    #     "lvis_v1_val_novel_rare_cats": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/only_rare/lvis_val_novel_shots.json")

    # },

    # "lvis_v1_fsod_rare":{
    #     "lvis_v1_fsod_train_novel_rare_cats": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/only_rare/lvis_train_novel_shots.json"),
    #     "lvis_v1_fsod_val_novel_rare_cats": ("/home/anishmad/msr_thesis/glip/DATASET/coco/", "/home/anishmad/msr_thesis/detic-lt3d/data/datasets/lvis/fsod/split/only_rare/lvis_val_novel_shots.json")

    # },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}

def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():               # e.g dataset_name is "lvis_v1_fsod"
        for key, (image_root, json_file) in splits_per_dataset.items():            # eg. key here is  "lvis_v1_fsod_train_novel"
            meta = get_lvis_instances_meta(dataset_name)
            register_lvis_instances(
                key,
                meta,
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/","cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_all_nuimages(_root)
    register_all_nuscenes(_root)
    register_all_nuimages_fsod(_root)