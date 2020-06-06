# Generating 3D Adversarial Point Clouds
By [Chong Xiang](https://nsec.sjtu.edu.cn/~chongxiang/), [Charles R. Qi](http://charlesrqi.com), [Bo Li](http://www.crystal-boli.com/home.html)

Sample code for CVPR 2019 paper "Generating 3D Adversarial Point Clouds" [arXiv](https://arxiv.org/abs/1809.07016)

<img src="https://github.com/xiangchong1/3d-adv-pc/blob/master/doc/attack_pipeline.png" width="50%" alt="attack pipeline" align=center>

## Requirement
This code is tested with Python 2.7 and Tensorflow 1.10.0

Other required packages include numpy, joblib, sklearn, etc. 

## Usage
There are four Python scripts in the root directorty for different attacks:
- perturbation.py -- Adversarial Point Pertubations
- independent.py -- Adversarial Independent Points
- cluster.py -- Adversarial Clusters
- object.py -- Adversarial Objects

The code logics of these four scripts are similar; they attack the victim objects into the specified target class.
The basic usage is `python perturbation.py --target=5`. 

Other parameters can be founded in the script, or run `python perturbation.py -h`. The default parameters are the ones used in the paper.



## Other files
- log/model.ckpt -- the victim model used in the paper. Download [link](https://drive.google.com/open?id=1T99mJfyuxFCcMQuvw71jgn6_FlUEOj08). 
- data/attacked_data.z -- the victim data used in the paper. It can be loaded with `joblib.load`, resulting in a Python list whose element is a numpy array (shape: 25\*1024\*3; 25 objects of the same class, each object is represented by 1024 points)
- **gen_initial.py** -- used to generate initial points for adversarial cluster/object. The script uses DBSCAN to cluster the generated critical points.
- critical -- the default directory to dump the generated initial points
- data/airplane.py -- the airplane object used in the paper as a uav for the adversarial object. can be loaded with ```np.load```.
- utils/tf_nndistance -- a self-defined tensorlfow op used for Chamfer/Hausdorff distance calculation. Use tf_nndistance_compile.sh to compile the op. The bash code may need modification according to the version and installtion path of CUDA. Note that it should be OK to directly calculate Chamfer/Hausdorff distance with available tf ops instead of tf_nndistance.

## Misc
- The sample adversarial point clouds can be downloaded [here](https://drive.google.com/open?id=1KLtJXFpq70YkB2DAxfUYyrWcv8kbkUJd). The targeted model is log/model.ckpt
- The aligned version of ModelNet40 data (in point cloud data format) can be downloaded [here](https://drive.google.com/open?id=1m7BmdtX1vWrpl9WRX5Ds2qnIeJHKmE36).
- The visulization in the paper is rendered with MeshLab
- Please open an issue or contact Chong Xiang (xiangchong97@gmail.com) if there is any question.
