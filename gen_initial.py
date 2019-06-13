import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import sys
from sklearn.cluster import DBSCAN
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_cluster', type=int, default=3, help='cluster number')

parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--critical_path', default='critical', help='the path to dump critical point')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')

#parameter for DBSCAN clustering
parser.add_argument('--max_num', type=int,help='max number of points selected from the critical point set for clustering',default=16)
parser.add_argument('--eps', type=float,default=0.2)
parser.add_argument('--min_num', type=int,help='the min number for each cluster',default=3)

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_CLUSTER = FLAGS.num_cluster
MODEL_PATH = FLAGS.model_path
CRIRICAL_PATH = FLAGS.critical_path
if not os.path.exists(CRIRICAL_PATH): os.mkdir(CRIRICAL_PATH)
DATA_DIR=FLAGS.data_dir
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

MAX_NUM = FLAGS.max_num
EPS = FLAGS.eps
MIN_NUM = FLAGS.min_num

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

def main():
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        vl=tf.global_variables()
        saver = tf.train.Saver(vl)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pre_max':end_points['pre_max'],
           'post_max':end_points['post_max']

    }

    for target in range(40):#40 is the number of classes
        clustered_cri_list=[]
        att_critical=get_critical_points_simple(sess,ops,attacked_data_all[target][:BATCH_SIZE])
        #joblib.dump(att_critical,os.path.join(BASE_DIR,CRIRICAL_PATH,'att_critical_{}.z' .format(target)))
        att_critical=[x[-MAX_NUM:,:] for x in att_critical]#get the points for DBSCAN clustering
        cri_all=np.concatenate(att_critical,axis=0)
        db = DBSCAN(eps=EPS, min_samples=MIN_NUM)
        result=db.fit_predict(cri_all)  # the cluster/class label of each point
        filter_idx=result > -0.5 #get the index of non-outlier point
        result=result[filter_idx]
        cri_all=cri_all[filter_idx]
        l,c=np.unique(result,return_counts=True)
        i_idx=np.argsort(c)[-NUM_CLUSTER:]
        l=l[i_idx]#get the label number for the largest NUM_CLUSTER clusters

        for label in l: 
            tmp=cri_all[result==label]#the point set belong to cluster "label"
            clustered_cri_list.append(tmp)
        joblib.dump(clustered_cri_list,os.path.join(BASE_DIR,CRIRICAL_PATH,'init_points_list_{}.z' .format(target)))

def get_critical_points_simple(sess,ops,data):

    #return all the critical points
    #note: this function is slightly different from get_critical_points() in the pointet_cls.py
    #the critical points returned by this function is a list, and the list elements are not in the same dimension
    #since all the critical points are returned

    is_training=False
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: is_training}
    pre_max_val,post_max_val=sess.run([ops['pre_max'],ops['post_max']],feed_dict=feed_dict)

    pre_max_val=np.reshape(pre_max_val,[BATCH_SIZE,NUM_POINT,1024])
    
    critical_points=[]
    for i in range(len(pre_max_val)):
        idx,counts=np.unique(np.argmax(pre_max_val[i],axis=0),return_counts=True)
        idx_idx=np.argsort(counts)
        idx=idx[idx_idx]
        points = data[i][idx]  
        critical_points.append(points)
    #critical_points=np.stack(critical_points)
    return critical_points


if __name__=='__main__':
    with tf.Graph().as_default():
        main()





