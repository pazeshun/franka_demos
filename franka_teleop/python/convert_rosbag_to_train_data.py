#!/usr/bin/env python
"""
Input: diretory that contain rosbag file for the same task.
Ouput: directory that has dataset for imitation lering
"""
import os
import sys
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm
from absl import app
from absl import flags
import yaml
import pickle
import pathlib2
import rospkg

try:
    import conversion_functions
except ImportError:
    sys.path.append(rospkg.RosPack().get_path('franka_teleop'))
    import conversion_functions


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'rosbag_dir', 'None', 'Directory that has target rosbag files')
flags.DEFINE_string(
    'target_dir', 'None', 'Target dir_name')
flags.DEFINE_string(
    'config', 'None', 'config file')

flags.DEFINE_integer(
    'im_size', 128, 'Resize image to (im_size, im_sizse)')


def check_and_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_img_msg(msg, save_file_name, size=(256, 256)):
    # TODO: fixed image size
    bridge = CvBridge()
    cv_img = bridge.compressed_imgmsg_to_cv2(msg)
    # w, h = cv_img.shape[0], cv_img.shape[1]
    # cv_img = cv_img[0:w, int((h-w)/2):int((h+w)/2)]
    cv_img = cv2.resize(cv_img, size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(save_file_name, cv_img)


def convert_rosbag(bag_file, target_dir, cfg_file):
    check_and_mkdir(target_dir)
    check_and_mkdir(os.path.join(target_dir, 'front_rgb'))  # TODO fix ahrd coding
    cfg = yaml.safe_load(open(cfg_file, 'rb'))
    img_count = 0
    data = {topic : [] for topic in cfg['topics'].keys()}
    save_topic = {topic : False for topic in cfg['topics'].keys()}    
    read_topics = cfg['topics'].keys() + ['/camera/rgb/image_raw/compressed']
    msg = 'cfg: {}'.format(cfg)
    with rosbag.Bag(bag_file, 'r') as bag:
        total_time = bag.get_end_time() - bag.get_start_time()
        print('converting {}...'.format(bag_file))
        print('total time {}'.format(total_time))
        for topic, msg, t in bag.read_messages(topics=read_topics):
            if topic == '/camera/rgb/image_raw/compressed':
                save_file_name = os.path.join(target_dir, "front_rgb", "{}.png".format(img_count))
                save_img_msg(msg, save_file_name)
                img_count += 1
                for tp in cfg['topics'].keys():
                    save_topic[tp] = True
            if topic in cfg['topics'].keys() and save_topic[topic]:
                raw_data = []
                for attr in cfg['topics'][topic]:
                    d = eval('msg.{}'.format(attr))
                    if type(d) == tuple:
                        d = list(d)
                    else:
                        d = [d]
                    raw_data += d
                data[topic].append(raw_data)
        # set trajectory steps to number of image by padding or cutting the last(s).
    for topic, d in data.items():
        if len(d) <= img_count:
            data[topic] = d + [d[-1]] * (img_count - len(data[topic]))
        if len(d) > img_count:
            data[topic] = data[topic][:img_count]
        assert len(data[topic])  == img_count, '{} has {} element but expected {}'.format(topic , len(data[topic]), img_count)
        data[topic] = np.array(data[topic])
    input_data = []
    output_data = []
    for rule in cfg['input']:
        # Not specifuying conversion function means identity conversion i.e. do nothing
        if len(rule) == 1:
            input_data.append(data[rule[0]])
        else:
            topic, func_name = rule
            func = getattr(conversion_functions, func_name)
            input_data.append(np.array([func(d) for d in data[topic]]))
    for rule in cfg['output']:
        # Not specifuying conversion function means identity conversion i.e. do nothing
        if len(rule) == 1:
            output_data.append(data[rule[0]])
        else:
            topic, func_name = rule
            func = getattr(conversion_functions, func_name)
            output_data.append(np.array([func(d) for d in data[topic]]))
    # input/output_data has now shape of (n_topics, seq_len, state_dim)
    # we convert it to (seq_len, state_dim_total)
    input_data = np.hstack(input_data)
    output_data = np.hstack(output_data)
    assert input_data.shape[0] == output_data.shape[0]
    print('seq_len, input_dim, outpu_dim : {}, {}, {}'.format(input_data.shape[0],
                                                              input_data.shape[1],
                                                              output_data.shape[1]))
    with open(os.path.join(target_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(target_dir, 'input_data.pkl'), 'wb') as f:
        pickle.dump(input_data, f)
    with open(os.path.join(target_dir, 'output_data.pkl'), 'wb') as f:
        pickle.dump(output_data, f)


def main(argv):
    target_dir_base = FLAGS.target_dir
    config_file = FLAGS.config
    p = pathlib2.Path(FLAGS.rosbag_dir)
    for i, f in enumerate(p.glob('*/rosbag/*bag')):
        print('opening {}th rosbag...'.format(i))
        f_lsit = f.as_posix().split('/')
        bag_count = f_lsit[-1][:3]
        variation = f_lsit[-3]
        target_dir = os.path.join(target_dir_base, 'default', variation, 'episodes', bag_count)
        convert_rosbag(f.as_posix(), target_dir, config_file)

if __name__ == '__main__':
    app.run(main)


