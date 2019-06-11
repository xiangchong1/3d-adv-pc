# 3d-adv-pc
Sample code for paper "Generating 3D Adversarial Point Clouds" [arXiv](https://arxiv.org/abs/1809.07016)

The code will be uploaded before the CVPR2019 conference!

![attack pipeline](https://github.com/xiangchong1/3d-adv-pc/master/doc/attack_pipeline)

## Requirement
This code is tested with Python 2.7 and Tensorflow 1.10.0

Other required packages include numpy, joblib, sklearn, etc. They can be installed with pip/conda.

## Usage
There are four Python scripts in the root directorty for different attacks:
- perturbation.py -- Adversarial Point Pertubations
- independent.py -- Adversarial Independent Points
- cluster.py -- Adversarial Clusters
- object.py -- Adversarial Objects

The code logics of these four scripts are similar; they attack the victim objects into the specified target class.
The basic usage is `python perturbation.py --target=5`. 

Other parameters can be founded in the script, or run `python perturbation -h`. The default parameters are the ones used in the paper.



## Other files
- log/model.ckpt -- the victim model used in the paper
- attacked_data.z -- the victim data used in the paper. It can be loaded with `joblib.load`, resulting in a Python list whose element is a numpy array (shape: 25\*1024\*3; 25 objects of the same class, each object is represented by 1024 points)
- **gen_initial.py** -- used to generate initial points for adversarial cluster/object. The script uses DBSCAN to cluster the generated critical points.
- critical -- the default directory to dump the generated initial points

- airplane.py -- the airplane object used in the paper as a uav for the adversarial object. can be loaded with ```np.load```.
- utils/tf_nndistance -- a self-defined tensorlfow op used for Chamfer/Hausdorff distance calculation. Use tf_nndistance_compile.sh to compile the op. The bash code may need modification according to the version and installtion path of CUDA. Note that it should be OK to directly calculate Chamfer/Hausdorff distance with available tf ops instead of tf_nndistance.

## Misc
- The sample adversarial point clouds can be downloaded [here](https://drive.google.com/drive/folders/1SCGNQVWDbpevv1f69MQNRvpYzVCheKt4?usp=sharing). The targeted model is log/model.ckpt
- The aligned version of ModelNet40 data (in point cloud data format) can be downloaded here (to be uploaded).
- The visulization in the paper is rendered with MeshLab
- Please contact Chong Xiang (xiangchong97@gmail.com) if there is any question.
