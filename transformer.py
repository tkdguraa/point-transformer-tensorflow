import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from pointnet_util import input_transform_net, pointnet_fp_module, sample_and_group


from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import knn_point, group_point

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def transformer_block(net, xyz, npoint, num, dim, is_training, bn_decay):

  print(net.shape)
  output_dim = dim

  net2 = tf.expand_dims(net, -2)
  net2 = tf_util.conv2d(net2, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='block1_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)
  net2 = tf.squeeze(net2)
  new_xyz, grouped_net2, grouped_xyz, gb_points = sample_and_group(npoint, 0, 16, xyz, net2)

  net2 = transformer(net2, xyz, grouped_xyz, gb_points, npoint, num, dim, is_training, bn_decay)
  net2 = tf_util.conv2d(net2, output_dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='block2_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)

  net2 = tf.squeeze(net2)


  return tf.concat([net, net2], -1)

def transformer(net_late, xyz, pos_enc, gb_points, npoint, num, dim, is_training, bn_decay):

  feature1 = net_late
  feature1 = tf.expand_dims(feature1, 2)
  last_dim = feature1.shape[-1]
  nsample = feature1.shape[-2]

  pos_enc = tf_util.conv2d(pos_enc, dim/2, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='pos_mlp1_%d_%d'%(npoint,num), bn_decay=bn_decay)
  beta = tf_util.conv2d(pos_enc, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='beta%d_%d'%(npoint,num), bn_decay=bn_decay, activation_fn=None, is_bias = False)


  phi = tf_util.conv2d(feature1, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='phi_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)

  print(xyz.shape, phi.shape, feature1.shape)
  _, phi, _, _ = sample_and_group(feature1.shape[1], 0, 16, xyz, tf.squeeze(phi))

  b_phi = tf_util.conv2d(feature1, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='b_phi_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)
  _, b_phi, _, _ = sample_and_group(feature1.shape[1], 0, 16, xyz, tf.squeeze(b_phi))


  alpha = tf_util.conv2d(feature1, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='alpha_%d_%d'%(npoint,num), activation_fn=None, is_bias=False, bn_decay=bn_decay)


  last_1 = tf_util.conv2d(b_phi - phi + beta, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='last1_%d_%d'%(npoint,num), bn_decay=bn_decay)

  last_1 = tf_util.conv2d(last_1, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='last2_%d_%d'%(npoint,num), activation_fn=None, is_bias=False, bn_decay=bn_decay)



  last_2 = beta + alpha
  last_1 = tf.nn.softmax(last_1, 2)
  edge_total = tf.multiply(last_1, last_2)
  edge_total = tf.reduce_sum(edge_total, -2, keep_dims=True)
  print("edge_total",edge_total.shape)

  return edge_total


def trasit_down(net, xyz, dim, num_points, num,is_training, bn_decay):
  new_xyz, net, grouped_xyz, gb_points = sample_and_group(num_points, 0, 8, xyz, net)
  net = tf_util.conv2d(net, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='transit_down_%d_%d'%(num_points, num),bn_decay=bn_decay)

  net = tf.reduce_max(net, -2)
 
  return new_xyz, net, grouped_xyz

def trasit_up(net, xyz, num_points, num,is_training, bn_decay):
  net = tf_util.conv2d(net2, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='transit_up1_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)
  net2 = tf_util.conv2d(net2, dim, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='transit_up2_%d_%d'%(npoint,num), activation_fn=None, is_bias=False,bn_decay=bn_decay)

###not completed

  return new_xyz, net, grouped_xyz



def get_model(point_cloud, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value
  feature_list = []
  end_points = {}
  xyz = point_cloud
  

  point_cloud = tf.expand_dims(point_cloud, -2)
  net = tf_util.conv2d(point_cloud, 64, [1, 1],
                       padding='VALID', stride=[1, 1],
                       bn=True, is_training=is_training,
                       scope='mlp1',bn_decay=bn_decay)
 
  net = tf.squeeze(net)  
  net = transformer_block(net, xyz, num_points, 0, 32, is_training, bn_decay)
  new_xyz, net, grouped_xyz = trasit_down(net, xyz, 32, num_points/2, 1, is_training, bn_decay)


  net = transformer_block(net, new_xyz, num_points/2, 1, 64, is_training, bn_decay)
  new_xyz, net, grouped_xyz = trasit_down(net, new_xyz, 64, num_points/4, 2, is_training, bn_decay)

  net = transformer_block(net, new_xyz, num_points/4, 2, 128, is_training, bn_decay)
  new_xyz, net, grouped_xyz = trasit_down(net, new_xyz, 128, num_points/8, 3, is_training, bn_decay)

  net = transformer_block(net, new_xyz, num_points/8, 3, 256, is_training, bn_decay)
  new_xyz, net, grouped_xyz = trasit_down(net, new_xyz, 256, num_points/16, 3, is_training, bn_decay)

  net = transformer_block(net, new_xyz, num_points/16, 4, 512, is_training, bn_decay)

  print("net=", net.shape)
  net = tf.reduce_max(net, 1)

  # MLP on global point cloud vector
  net = tf.reshape(net, [batch_size, -1]) 
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

  return net, end_points, feature_list


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)


  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)
