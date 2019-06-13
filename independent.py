import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import tf_nndistance
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='independent', help='dump folder path [independent]')

parser.add_argument('--add_num', type=int, default=512, help='number of added points [default: 512]')
parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--constraint', default='c', help='type of constraint. h for Hausdoff; c for Chamfer')
parser.add_argument('--lr_attack', type=float, default=0.01, help='learning rate for optimization based attack')

parser.add_argument('--initial_weight', type=float, default=5000, help='initial value for the parameter lambda')#200 for Hausdorff
parser.add_argument('--upper_bound_weight', type=float, default=40000, help='upper_bound value for the parameter lambda')#900 for Hausdorff
parser.add_argument('--step', type=int, default=10, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(FLAGS.log_dir, "model.ckpt")
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DATA_DIR = FLAGS.data_dir

TARGET=FLAGS.target
NUM_ADD=FLAGS.add_num
LR_ATTACK=FLAGS.lr_attack
#WEIGHT=FLAGS.weight

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

INITIAL_WEIGHT=FLAGS.initial_weight
UPPER_BOUND_WEIGHT=FLAGS.upper_bound_weight
#ABORT_EARLY=False
BINARY_SEARCH_STEP=FLAGS.step
NUM_ITERATIONS=FLAGS.num_iter

def attack():
    is_training = False
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_ADD,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
            initial_point_pl=tf.placeholder(shape=[BATCH_SIZE,NUM_ADD,3],dtype=tf.float32)
            point_added=initial_point_pl+pert
            pointclouds_input=tf.concat([pointclouds_pl,point_added],axis=1)
            
            pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)

            #adv loss
            adv_loss=MODEL.get_adv_loss(pred,TARGET)
               
            dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(point_added,pointclouds_pl)
            if FLAGS.constraint=='c':#Chamfer 
                dists_forward=tf.reduce_mean(dists_forward,axis=1)
                dists_backward=tf.reduce_mean(dists_backward,axis=1)#not used
            elif FLAGS.constraint=='h':#Hausdorff
                dists_forward=tf.reduce_max(dists_forward,axis=1)
                dists_backward=tf.reduce_max(dists_backward,axis=1)#not used
            else:
                raise Exception("Invalid constraint type. Please try c for Chamfer and h for Hausdorff")

            dist_weight=tf.placeholder(shape=[BATCH_SIZE],dtype=tf.float32)
            lr_attack=tf.placeholder(dtype=tf.float32)
            attack_optimizer = tf.train.AdamOptimizer(lr_attack)
            attack_op = attack_optimizer.minimize(adv_loss + tf.reduce_mean(tf.multiply(dist_weight,dists_forward)),var_list=[pert])
            
            vl=tf.global_variables()
            vl=[x for x in vl if 'pert' not in x.name]
            saver = tf.train.Saver(vl)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pointclouds_input':pointclouds_input,
               'initial_point_pl':initial_point_pl,
               'dist_weight':dist_weight,
               'pert': pert,
               'point_added':point_added,
               'pre_max':end_points['pre_max'],
               'post_max':end_points['post_max'],
               'pred': pred,
               'adv_loss': adv_loss,
               #'dists_backward':dists_backward,
               'dists_forward':dists_forward,
               'total_loss':tf.reduce_mean(tf.multiply(dist_weight,dists_forward))+adv_loss,
               'lr_attack':lr_attack,
               'attack_op':attack_op
               }

        saver.restore(sess,MODEL_PATH)
        print('model restored!')

        dist_list=[]  
        for victim in [5,35,33,22,37,2,4,0,30,8]:#the class index of selected 10 largest classed in ModelNet40
            if victim == TARGET:
                continue
            attacked_data=attacked_data_all[victim]#attacked_data shape:25*1024*3
            for j in range(25//BATCH_SIZE):
                dist,img=attack_one_batch(sess,ops,attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
                dist_list.append(dist)
                np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_adv.npy' .format(victim,TARGET,j)),img)
                #np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_orig.npy' .format(victim,TARGET,j)),attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])#dump originial example for comparison
        #joblib.dump(dist_list,os.path.join('.',DUMP_DIR,'dist_{}.z' .format(TARGET)))#log distance information for performation evaluation
def attack_one_batch(sess,ops,attacked_data):

    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'attacked_data' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = False

    attacked_label=np.ones(shape=(len(attacked_data)),dtype=int) * TARGET #the target label for adv pcs
    attacked_label=np.squeeze(attacked_label)

    #the bound for the binary search
    lower_bound=np.zeros(BATCH_SIZE)
    WEIGHT = np.ones(BATCH_SIZE) * INITIAL_WEIGHT
    upper_bound=np.ones(BATCH_SIZE) * UPPER_BOUND_WEIGHT

   
    o_bestdist = [1e10] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_ADD+NUM_POINT,3))

    init_points=MODEL.get_critical_points(sess,ops,attacked_data,BATCH_SIZE,NUM_ADD,NUM_POINT)

    feed_dict = {ops['pointclouds_pl']: attacked_data,
         ops['is_training_pl']: is_training,
         ops['lr_attack']:LR_ATTACK,
         ops['dist_weight']:WEIGHT,
         ops['initial_point_pl']:init_points}

    for out_step in range(BINARY_SEARCH_STEP):

        feed_dict[ops['dist_weight']]=WEIGHT

        sess.run(tf.assign(ops['pert'],tf.truncated_normal([BATCH_SIZE,NUM_ADD,3], mean=0, stddev=0.0000001)))

        bestdist = [1e10] * BATCH_SIZE
        bestscore = [-1] * BATCH_SIZE  

        prev = 1e6      

        for iteration in range(NUM_ITERATIONS):
            _= sess.run([ops['attack_op']], feed_dict=feed_dict)

            adv_loss_val,dist_val,pred_val,input_val = sess.run([ops['adv_loss'],
                ops['dists_forward'],ops['pred'],ops['pointclouds_input']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            loss=adv_loss_val+np.average(dist_val*WEIGHT)
            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                print((" Iteration {} of {}: loss={} adv_loss:{} " +
                               "distance={}")
                              .format(iteration, NUM_ITERATIONS,
                                loss, adv_loss_val,np.mean(dist_val)))


            # check if we should abort search if we're getting nowhere.
            '''
            if ABORT_EARLY and iteration % ((MAX_ITERATIONS // 10) or 1) == 0:
                
                if loss > prev * .9999999:
                    msg = "    Failed to make progress; stop early"
                    print(msg)
                    break
                prev = loss
            '''

            for e, (dist, pred, ii) in enumerate(zip(dist_val, pred_val, input_val)):
                if dist < bestdist[e] and pred == TARGET:
                    bestdist[e] = dist
                    bestscore[e] = pred
                if dist < o_bestdist[e] and pred == TARGET:
                    o_bestdist[e]=dist
                    o_bestscore[e]=pred
                    o_bestattack[e] = ii

        # adjust the constant as needed
        for e in range(BATCH_SIZE):
            if bestscore[e]==TARGET and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e] :
                # success
                lower_bound[e] = max(lower_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
                #print('new result found!')
            else:
                # failure
                upper_bound[e] = min(upper_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
        #bestdist_prev=deepcopy(bestdist)

    print(" Successfully generated adversarial exampleson {} of {} instances." .format(sum(lower_bound > 0), BATCH_SIZE))
    return o_bestdist,o_bestattack


if __name__=='__main__':
    attack()
