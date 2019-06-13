import numpy as np 
import tensorflow as tf

def shift_object(sh,center,num_point,scale):
    def normalize(data,scale):
        center=(np.max(data,axis=0)+np.min(data,axis=0))/2
        data=data - np.expand_dims(center,axis=0)
        norm = np.linalg.norm(data,axis=1)
        radius=np.max(norm)
        data=data/radius
        data=data*scale
        return data
    sh=normalize(sh,scale)
    if sh.shape[0] > num_point:
        np.random.shuffle(sh)
        sh=sh[:num_point]
    center=np.array(center)
    center=np.reshape(center,[1,3])
    return sh+center

def euler2mat_tf(point_cloud,rotations):
    batch_size = rotations.get_shape()[0].value
    assert rotations.get_shape()[1].value == 3
    rotated_list=[]
    one=tf.constant([1.])
    zero=tf.constant([0.])
    #print(zero.get_shape())
    for i in range(batch_size):
        x=rotations[i,0]
        y=rotations[i,1]
        z=rotations[i,2]
        cosz = tf.cos([z])
        sinz = tf.sin([z])
        #print(cosz.get_shape())
        Mz=tf.stack(
                [[cosz, -sinz, zero],
                 [sinz, cosz, zero],
                 [zero, zero, one]])
        Mz=tf.squeeze(Mz)
        cosy = tf.cos([y])
        siny = tf.sin([y])
        My=tf.stack(
                [[cosy, zero, siny],
                 [zero, one,zero],
                 [-siny, zero, cosy]])
        My=tf.squeeze(My)
        cosx = tf.cos([x])
        sinx = tf.sin([x])
        Mx=tf.stack(
                [[one,zero, zero],
                 [zero, cosx, -sinx],
                 [zero, sinx, cosx]])
        Mx=tf.squeeze(Mx)
        rotate_mat=tf.matmul(Mz,tf.matmul(My,Mz))
        rotated_list.append(tf.matmul(point_cloud[i],rotate_mat))

    return tf.stack(rotated_list)