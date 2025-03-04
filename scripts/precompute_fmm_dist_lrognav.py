import _pickle as cPickle
import bz2
import multiprocessing as mp
import os

import tqdm
from poni.dataset import SemanticMapDataset
from poni.default import get_cfg


# assert "ACTIVE_DATASET" in os.environ
# ACTIVE_DATASET = os.environ["ACTIVE_DATASET"]
ACTIVE_DATASET = "gibson"

SEED = 123
DATA_ROOT = "data/semantic_maps/{}/semantic_maps".format(ACTIVE_DATASET)
print("DATA_ROOT=",DATA_ROOT)
SAVE_ROOT = "data/semantic_maps/{}/fmm_dists_{}".format(ACTIVE_DATASET, SEED)
SAVE_ROOT_ROOM = "data/semantic_maps/{}/fmm_dists_{}".format(ACTIVE_DATASET, SEED)

print("SAVE_ROOT=",SAVE_ROOT)
NUM_WORKERS = 24


def save_data(inputs):
    data, path = inputs
    # print("inputs=",inputs)
    with bz2.BZ2File(path, "w") as fp:
        cPickle.dump(data, fp)

def save_data_room(inputs):
    data, path = inputs
    # print("inputs=",inputs)
    with bz2.BZ2File(path, "w") as fp:
        cPickle.dump(data, fp)

def precompute_fmm_dists():
    cfg = get_cfg()
    # print("cfg=",cfg)
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = ACTIVE_DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.fmm_dists_saved_root = ""
    cfg.freeze()
    os.makedirs(SAVE_ROOT, exist_ok=True)
    pool = mp.Pool(NUM_WORKERS)

    for split in ["val", "train"]:
        print("split=",split)
        print(f"=====> Computing FMM dists for {split} split")
        dataset = SemanticMapDataset(cfg.DATASET, split=split)
        # print("cfg.DATASET=",cfg.DATASET)
        print("dataset=",dataset)
        print("--> Saving FMM dists")
        inputs = []
        for name in dataset.names:
            print("dataset.names=",dataset.names)
            save_path = os.path.join(SAVE_ROOT, f"{name}.pbz2")
            data = dataset.fmm_dists[name]
            print("data=",data[0].shape)
            inputs.append((data, save_path))
        _ = list(tqdm.tqdm(pool.imap(save_data, inputs), total=len(inputs)))
        # print("_ =",_)



def precompute_fmm_dists_room():
    cfg = get_cfg()
    # print("cfg=",cfg)
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = ACTIVE_DATASET
    cfg.DATASET.root = "/home/aae14859ln/Sun/PONI/data/semantic_maps/gibson/semantic_maps"
    cfg.DATASET.fmm_dists_saved_root = ""
    cfg.freeze()
    os.makedirs(SAVE_ROOT_ROOM, exist_ok=True)
    pool = mp.Pool(NUM_WORKERS)

    for split in ["val", "train"]:
        print("split=",split)
        print(f"=====> Computing FMM dists for {split} split")
        dataset = SemanticMapDataset(cfg.DATASET, split=split)
        # print("cfg.DATASET=",cfg.DATASET)
        print("dataset=",dataset)
        print("--> Saving FMM dists")
        inputs = []
        for name in dataset.names:
            print("dataset.names=",dataset.names)
            save_path = os.path.join(SAVE_ROOT_ROOM, f"{name}.pbz2")
            data = dataset.fmm_dists[name]
            print("data=",data[0].shape)
            inputs.append((data, save_path))
        _ = list(tqdm.tqdm(pool.imap(save_data_room, inputs), total=len(inputs)))


if __name__ == "__main__":
    precompute_fmm_dists()
    precompute_fmm_dists_room()

