import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import tf_nndistance
import joblib
from shift_rotate_util import euler2mat_tf,shift_object

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='object', help='dump folder path [object]')

parser.add_argument('--add_num', type=int, default=64, help='number of added points [default: 512]')
parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--lr_attack', type=float, default=0.01, help='learning rate for optimization based attack')

parser.add_argument('--initial_weight', type=float, default=5, help='initial value for the parameter lambda')
parser.add_argument('--upper_bound_weight', type=float, default=40, help='upper_bound value for the parameter lambda')
parser.add_argument('--step', type=int, default=5, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')
parser.add_argument('--mu', type=float, default=0.2, help='preset value for parameter mu')
parser.add_argument('--init_dir', default='critical', help='the dir which contains the initial point')

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
MU=FLAGS.mu
INIT_PATH=FLAGS.init_dir

#put the following code in the main code script
assert os.path.exists(os.path.join(BASE_DIR,INIT_PATH,'init_points_list_{}.z' .format(TARGET))), 'No init point found! run dbscan_clustering.py to generate init points'

init_points_list=joblib.load(os.path.join(BASE_DIR,INIT_PATH,'init_points_list_{}.z' .format(TARGET)))

NUM_CLUSTER=len(init_points_list)#sometimes, there is only a limited number of cluster formed
                                 #so that DBSCAN may only get a NUM_CLUSTER smaller than the specified parameter
                                 #considering that, NUM_CLUSTER in this script is not a given parameter but obtained from the init point data

#make sure each element in init_point_list is in shape of BATCH_SIZE*NUM_ADD*3

airplane=np.load(os.path.join(DATA_DIR,'airplane.npy'))

for i in range(NUM_CLUSTER):
    tmp=init_points_list[i]
    cls_center=np.mean(tmp,axis=0)
    tmp=shift_object(airplane,cls_center,FLAGS.add_num,0.3)
    tmp=np.expand_dims(tmp,axis=0)
    init_points_list[i]=np.tile(tmp,[BATCH_SIZE,1,1])


def attack():
    is_training = False
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            point_added_list=[]
            pert_list=[]
            initial_point_pl_list=[]
            adv_scale_list=[]
            nndistance_loss_list=[]

            for i in range(NUM_CLUSTER):
                pert=tf.get_variable(name='pert_{}' .format(i),shape=[BATCH_SIZE,NUM_ADD+2,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
                shift=pert[:,-1,:]
                rotation=pert[:,-2,:]
                pert_added=pert[:,:NUM_ADD,:]

                initial_point_pl=tf.placeholder(name='init_pl_{}'.format(i),shape=[BATCH_SIZE,NUM_ADD,3],dtype=tf.float32)
                shift=tf.expand_dims(shift,axis=1)
                point_added=initial_point_pl+pert_added+shift
                point_added=euler2mat_tf(point_added,rotation)
                point_added_list.append(point_added)
                pert_list.append(pert)
                initial_point_pl_list.append(initial_point_pl)

                #farthest distance loss 

                adv_scale=tf.sqrt(tf.reduce_sum(tf.square(pert_added),[1,2]))
                adv_scale_list.append(adv_scale)

                #Chamfer/Hausdorff
                dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(point_added,pointclouds_pl)
                dists_forward=tf.reduce_mean(dists_forward,axis=1)#Chamfer
                nndistance_loss_list.append(dists_forward)

            pointclouds_input=tf.concat([pointclouds_pl]+point_added_list,axis=1)
            
            pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)

            #adv loss
            adv_loss=MODEL.get_adv_loss(pred,TARGET)
               
            dist_weight=tf.placeholder(shape=[BATCH_SIZE],dtype=tf.float32)
            lr_attack=tf.placeholder(dtype=tf.float32)
            attack_optimizer = tf.train.AdamOptimizer(lr_attack)
            attack_op_list=[]
            for i in range(NUM_CLUSTER):
                attack_op = attack_optimizer.minimize(adv_loss + 
                    tf.reduce_mean(tf.multiply(adv_scale_list[i]+MU*nndistance_loss_list[i],dist_weight)),
                    var_list=[pert_list[i]])
                attack_op_list.append(attack_op)
            
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
               'initial_point_pl_list':initial_point_pl_list,
               'pert_list': pert_list,
               'point_added_list':point_added_list,
               'dist_weight':dist_weight,
               'point_added':point_added,
               'pre_max':end_points['pre_max'],
               'post_max':end_points['post_max'],
               'pred': pred,
               'adv_loss': adv_loss,
               'adv_scale_list':adv_scale_list,
               'nndistance_loss_list':nndistance_loss_list,
               'lr_attack':lr_attack,
               'attack_op_list':attack_op_list
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
    o_bestdist_c= [1e10] * BATCH_SIZE
    o_bestdist_p = [1e10] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_ADD*NUM_CLUSTER+NUM_POINT,3))
 
    feed_dict = {ops['pointclouds_pl']: attacked_data,
     ops['is_training_pl']: is_training,
     ops['lr_attack']:LR_ATTACK}

    for j in range(NUM_CLUSTER):
        feed_dict[ops['initial_point_pl_list'][j]]=init_points_list[j]

    for out_step in range(BINARY_SEARCH_STEP):

        feed_dict[ops['dist_weight']]=WEIGHT
        for j in range(NUM_CLUSTER):
            sess.run(tf.assign(ops['pert_list'][j],tf.truncated_normal([BATCH_SIZE,NUM_ADD+2,3], mean=0, stddev=0.0000001)))

        bestdist = [1e10] * BATCH_SIZE
        bestscore = [-1] * BATCH_SIZE  

        prev = 1e6      

        for iteration in range(NUM_ITERATIONS):
            for j in range(NUM_CLUSTER):
                _= sess.run([ops['attack_op_list'][j]], feed_dict=feed_dict)

            adv_loss_val,pred_val,input_val = sess.run([ops['adv_loss'],
                    ops['pred'],ops['pointclouds_input']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            adv_scale_list_val=sess.run(ops['adv_scale_list'],feed_dict)
            adv_scale_list_val=np.stack(adv_scale_list_val)
            adv_scale_list_val=np.average(adv_scale_list_val,axis=0)

            nndistance_loss_list_val=sess.run(ops['nndistance_loss_list'],feed_dict)
            nndistance_loss_list_val=np.stack(nndistance_loss_list_val)
            nndistance_loss_list_val=np.average(nndistance_loss_list_val,axis=0)    

            loss=adv_loss_val+np.average((adv_scale_list_val+MU*nndistance_loss_list_val)*WEIGHT)
          
            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                print((" Iteration {} of {}: loss={} adv_loss:{} " +
                               "distance={},{}")
                              .format(iteration, NUM_ITERATIONS,
                                loss, adv_loss_val,np.average(adv_scale_list_val),np.average(nndistance_loss_list_val)))


            # check if we should abort search if we're getting nowhere.
            '''
            if ABORT_EARLY and iteration % ((MAX_ITERATIONS // 10) or 1) == 0:
                
                if loss > prev * .9999999:
                    msg = "    Failed to make progress; stop early"
                    print(msg)
                    break
                prev = loss
            '''

            for e, (dist_c,dist_p,pred, ii) in enumerate(zip(nndistance_loss_list_val,adv_scale_list_val, pred_val, input_val)):
                dist=dist_c*MU+dist_p
                if dist < bestdist[e] and pred == TARGET:
                    bestdist[e] = dist
                    bestscore[e] = pred
                if dist < o_bestdist[e] and pred == TARGET:
                    o_bestdist[e]=dist
                    o_bestdist_c[e]=dist_c
                    o_bestdist_p[e]=dist_p
                    o_bestscore[e]=pred
                    o_bestattack[e] = ii

        # adjust the weight as needed
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
    return [o_bestdist,o_bestdist_p,o_bestdist_c],o_bestattack


if __name__=='__main__':
    attack()
