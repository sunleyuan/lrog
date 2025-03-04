
## Installation

Clone the current repo and required submodules:
```
git clone git@github.com:srama2512/PONI.git
cd PONI
git submodule init
git submodule update
export PONI_ROOT=<PATH TO PONI/>
```
 Create a conda environment:
```
conda create --name poni python=3.8.5
conda activate poni
```

Install pytorch (assuming cuda 10.2):
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
```

Install dependencies:
```
cd $PONI_ROOT/dependencies/habitat-lab
pip install -r requirements.txt
python setup.py develop --all

cd $PONI_ROOT/dependencies/habitat-sim
pip install -r requirements.txt
python setup.py install --headless --with-cuda

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu102.html

cd $PONI_ROOT/dependencies/astar_pycpp && make
```

Install requirements for PONI:
```
cd $PONI_ROOT
pip install -r requirements.txt
```

Add repository to python path:
```
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT
```


## Creating semantic map datasets

1. Download [Gibson](http://gibsonenv.stanford.edu/database/) and [Matterport3D](https://niessner.github.io/Matterport/) scenes following the instructions [here](DATASETS.md).

2. Extract Gibson semantic maps.
    ```
    cd $PONI_ROOT
    ACTIVE_DATASET="gibson" python scripts/create_semantic_maps_lrognav.py
    ```

3. Create dataset for PONI training. </br>
    a. First extract FMM distances for all objects in each map.
    ```
    cd $PONI_ROOT
    ACTIVE_DATASET="gibson" python scripts/precompute_fmm_dists_lrognav.py
    ```
    b. Extract training and validation data for PONI.
    ```
    ACTIVE_DATASET="gibson" python scripts/create_poni_dataset_lrognav.py --split "train"
    ACTIVE_DATASET="gibson" python scripts/create_poni_dataset_lrognav.py --split "val"
    ```
4. The extracted data can be visualized using [notebooks/visualize_pfs.ipynb](notebooks/visualize_pfs.ipynb).
5. The `create_poni_dataset.py` script also supports parallelized dataset creation. The `--map-id` argument can be used to limit the data generation to one specific map. The `--map-id-range` argument can be used to limit the data generation to maps in range `i` to `j` as follows: `--map-id-range i j`. These arguments can be used to divide the data generation across multiple processes within a node or on a cluster with SLURM by passing the appropriate map ids to each job.


## Training

To train models for PONI, predict-xy, predict-theta, and predict-action methods, copy over corresponding scripts from `$PONI_ROOT/experiment_scripts/<DATASET_NAME>/train_<METHOD_NAME>.sh` to some experiment directory and execute it. For example, to train PONI on Gibson:
```
mkdir -p $PONI_ROOT/experiments/poni/
cd $PONI_ROOT/experiments/poni
cp $PONI_ROOT/experiment_scripts/gibson/train_poni.sh .
chmod +x train_poni.sh
./train_poni.sh
```


## ObjectNav evaluation on Gibson

We use a modified version of the Gibson ObjectNav evaluation setup from [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation).

1. Download the [Gibson ObjectNav dataset](https://utexas.box.com/s/tss7udt3ralioalb6eskj3z3spuvwz7v) to `$PONI_ROOT/data/datasets/objectnav/gibson`.
    ```
    cd $PONI_ROOT/data/datasets/objectnav
    wget -O gibson_objectnav_episodes.tar.gz https://utexas.box.com/shared/static/tss7udt3ralioalb6eskj3z3spuvwz7v.gz
    tar -xvzf gibson_objectnav_episodes.tar.gz && rm gibson_objectnav_episodes.tar.gz
    ```
2. Download the image segmentation model [[URL](https://utexas.box.com/s/sf4prmup4fsiu6taljnt5ht8unev5ikq)] to `$PONI_ROOT/pretrained_models`.
3. Copy the evaluation script corresponding to the model of interest from `$PONI_ROOT/experiment_scripts/gibson/eval_<METHOD_NAME>.sh` to the required experiment directory. 
5. Set the `MODEL_PATH` variable in the script to the saved checkpoint. By default, it points to the path of a pre-trained model (see previous section).
5. To reproduce results from the paper, download the pre-trained models and evaluate them using the evaluation scripts.
6. To visualize episodes with the semantic map and potential function predictions, add the arguments `--print_images 1 --num_pf_maps 3` in the evaluation script.



## Acknowledgements

In our work, we used parts of [Semantic-MapNet](https://github.com/vincentcartillier/Semantic-MapNet), [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), [Object-Goal-Navigation](https://github.com/devendrachaplot/Object-Goal-Navigation), [astar_pycpp](https://github.com/srama2512/astar_pycpp) and [PONI](https://github.com/srama2512/PONI) repos and extended them.


## License
This project is released under the MIT license, as found in the [LICENSE](LICENSE) file.

