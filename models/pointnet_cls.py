import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import tf_nndistance
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    #print("before maxpool")
    #print(net.get_shape())
    end_points['pre_max']=net
    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    end_points['post_max']=net
    #print("after maxpool")
    #print(net.get_shape())
    net = tf.reshape(net, [batch_size, -1])
    #print("after reshape")
    #print(net.get_shape())
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    #print(end_points['pre_max'].get_shape())
    return net, end_points


def get_adv_loss(unscaled_logits,targets,kappa=0):
    
    with tf.variable_scope('adv_loss'):
        unscaled_logits_shape = tf.shape(unscaled_logits)

        B = unscaled_logits_shape[0]
        K = unscaled_logits_shape[1]

        tlab=tf.one_hot(targets,depth=K,on_value=1.,off_value=0.)
        tlab=tf.expand_dims(tlab,0)
        tlab=tf.tile(tlab,[B,1])
        real = tf.reduce_sum((tlab) * unscaled_logits, 1)
        other = tf.reduce_max((1 - tlab) * unscaled_logits -
                              (tlab * 10000), 1)
        loss1 = tf.maximum(np.asarray(0., dtype=np.dtype('float32')), other - real + kappa)
        return tf.reduce_mean(loss1)

def get_critical_points(sess,ops,data,BATCH_SIZE,NUM_ADD,NUM_POINT=1024):

    ####################################################
    ### get the critical point of the given point clouds
    ### data shape: BATCH_SIZE*NUM_POINT*3
    ### return : BATCH_SIZE*NUM_ADD*3
    #####################################################
    sess.run(tf.assign(ops['pert'],tf.zeros([BATCH_SIZE,NUM_ADD,3])))
    is_training=False

    #to make sure init_points is in shape of BATCH_SIZE*NUM_ADD*3 so that it can be fed to initial_point_pl
    if NUM_ADD > NUM_POINT:
        init_points=np.tile(data[:,:2,:],[1,NUM_ADD/2,1]) ## due to the max pooling operation of PointNet, 
                                                          ## duplicated points would not affect the global feature vector   
    else:
        init_points=data[:, :NUM_ADD, :]
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: is_training,
                 ops['initial_point_pl']:init_points}
    pre_max_val,post_max_val=sess.run([ops['pre_max'],ops['post_max']],feed_dict=feed_dict)
    pre_max_val = pre_max_val[:,:NUM_POINT,...]
    pre_max_val=np.reshape(pre_max_val,[BATCH_SIZE,NUM_POINT,1024])#1024 is the dimension of PointNet's global feature vector
    
    critical_points=[]
    for i in range(len(pre_max_val)):
        #get the most important critical points if NUM_ADD < number of critical points
        #the importance is demtermined by counting how many elements in the global featrue vector is 
        #contributed by one specific point 
        idx,counts=np.unique(np.argmax(pre_max_val[i],axis=0),return_counts=True)
        idx_idx=np.argsort(counts)
        if len(counts) > NUM_ADD:
            points = data[i][idx[idx_idx[-NUM_ADD:]]]
        else:
            points = data[i][idx]
            tmp_num = NUM_ADD - len(counts)
            while(tmp_num > len(counts)):
                points = np.concatenate([points,data[i][idx]])
                tmp_num-=len(counts)
            points = np.concatenate([points,data[i][-tmp_num:]])
        
        critical_points.append(points)
    critical_points=np.stack(critical_points)
    return critical_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        inputs2 = tf.zeros((32,122,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(inputs2,inputs)
