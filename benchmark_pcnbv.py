import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util_nbv import lmdb_dataflow, get_queued_data
from termcolor import colored
import pdb
from tensorpack import dataflow
from scipy import stats
import csv
import envs.pc_nbv_env
import logging
import numpy as np
# nohup python benchmark_pcnbv.py > benchmark.log 2>&1 &


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs') # input point cloud
    npts_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points') 
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths') # ground truth
    view_state_pl = tf.placeholder(tf.float32, (args.batch_size, args.views), 'view_state') # view space selected state
    eval_value_pl = tf.placeholder(tf.float32, (args.batch_size, args.views, 1), 'eval_value') # surface cov, 

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, view_state_pl, eval_value_pl, is_training = is_training_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, args.checkpoint)
    print(colored('Testing...', 'grey', 'on_green'))
    sess.run(tf.local_variables_initializer())
    
    # logging
    logger = logging.getLogger("./log_benchmark.log")
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    shell_handle = logging.StreamHandler()
    shell_handle.setFormatter(log_format)
    shell_handle.setLevel(logging.DEBUG)
    logger.addHandler(shell_handle)
    # create environment
    test_env = envs.pc_nbv_env.PointCloudNextBestViewEnv(data_path=args.test_data_path,
                                                            view_num=args.view_num,
                                                            observation_space_dim=args.observation_space_dim,
                                                            log_level=logging.ERROR)
    model_size = test_env.shapenet_reader.model_num
    logger.info("model size: {}".format(model_size))
    init_step = 0
    average_coverage = np.zeros(args.step_size)
    cur_npts = np.asarray([1024], dtype=np.int32)
    cur_eval_value = np.zeros((1, 33, 1), dtype=np.float64)
    for model_id in range(model_size):
        obs = test_env.reset(init_step=init_step)
        cur_gt = test_env.get_gt()
        cur_model_name = test_env.shapenet_reader.cur_model_name
        logger.info("handle model {}: {}".format(model_id, cur_model_name))
        init_step = (init_step + 1) % args.view_num
        average_coverage[0] += test_env.current_coverage
        for step_id in range(args.step_size - 1):
            cur_view_state = obs["view_state"]
            cur_input = obs["current_point_cloud"]
            feed_dict = {inputs_pl: cur_input, npts_pl: cur_npts, gt_pl: cur_gt, view_state_pl:cur_view_state, 
                eval_value_pl:cur_eval_value, is_training_pl: False}
            test_loss, test_loss_eval, test_eval_value_pre = sess.run([model.loss, model.loss_eval, model.eval_value], feed_dict=feed_dict)
            action = 0
            max_score = 0
            for i in range(33):
                if test_eval_value_pre[0][i][0] > max_score:
                    action = i
                    max_score = test_eval_value_pre[0][i][0]
            obs, rewards, dones, info = test_env.step(action)
            average_coverage[step_id + 1] += info["current_coverage"]
            if (step_id == args.step_size - 2) and (info["current_coverage"] <= 0.9):
                logger.error("model name: {}, step: {}, coverage: {}".format(test_env.model_name, args.step_size, info["current_coverage"]))
    average_coverage = average_coverage / model_size
    average_coverage = average_coverage * 100
    
    with open("average_coverage.txt", "a+", encoding="utf-8") as f:
        f.write("{}: ".format("PC-NBV"))
        for coverage in average_coverage:
            f.write("{:.2f}, ".format(coverage))
        f.write("\n")

    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_test', default='/root/tf/test.lmdb')
    parser.add_argument('--model_type', default='pc-nbv')
    parser.add_argument('--checkpoint', default='/root/tf/PC-NBV/log/6_13/model-400000')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=512)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--views', type=int, default=33)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--view_num', type=int, default=33)
    parser.add_argument('--test_data_path', type=str, default="/root/tf/benchmark_data/test_bak")
    parser.add_argument('--observation_space_dim', type=int, default=1024)
    parser.add_argument('--step_size', type=int, default=10)

    args = parser.parse_args()

    train(args)