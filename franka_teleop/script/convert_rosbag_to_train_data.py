#!/usr/bin/env python
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm
from absl import app
from absl import flags
import yaml
import pickle
import pathlib2


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


def save_img_msg(msg, save_file_name, size=(128, 128)):
    # TODO: fixed image size
    bridge = CvBridge()
    cv_img = bridge.compressed_imgmsg_to_cv2(msg)
    # w, h = cv_img.shape[0], cv_img.shape[1]
    # cv_img = cv_img[0:w, int((h-w)/2):int((h+w)/2)]
    cv_img = cv2.resize(cv_img, size, interpolation = cv2.INTER_AREA)
    cv2.imwrite(save_file_name, cv_img)


def convert_one_rosbag(bag_file, target_dir, cfg_file):
    # bag_file = FLAGS.rosbag
    # target_dir = FLAGS.target_dir
    check_and_mkdir(target_dir)
    check_and_mkdir(os.path.join(target_dir, 'front_rgb'))  # TODO fix ahrd coding
    cfg = yaml.safe_load(open(cfg_file, 'rb'))
    img_count = 0
    input_data = {}
    output_data = {}
    save_topic = {}
    for topic in cfg['input']:
        input_data[topic] = []
    for topic in cfg['output']:
        output_data[topic] = []
    for tp in cfg['topics'].keys():
        save_topic[tp] = False
    read_topics = cfg['topics'].keys() + ['/camera/rgb/image_raw/compressed']
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
                if topic in cfg['input']:
                    input_data[topic].append(raw_data)
                if topic in cfg['output']:
                    output_data[topic].append(raw_data)
                save_topic[tp] = False
        for k, v in input_data.items():
            if len(v) < img_count:
                input_data[k] += v[-1] * (img_count - len(v))
            if len(v) > img_count:
                input_data[k] = input_data[k][:img_count]

        for k, v in output_data.items():
            if len(v) < img_count:
                output_data[k] += v[-1] * (img_count - len(v))
            if len(v) > img_count:
                output_data[k] = output_data[k][:img_count]
        for k, v in input_data.items():
            assert len(v) == img_count, '{} has {} element but expected {}'.format(k , len(v), img_count)
        for k, v in output_data.items():
            assert len(v) == img_count, '{} has {} element but expected {}'.format(k , len(v), img_count)
        with open(os.path.join(target_dir, 'input_data.pkl'), 'wb') as f:
            pickle.dump(input_data, f)
        with open(os.path.join(target_dir, 'output_data.pkl'), 'wb') as f:
            pickle.dump(output_data, f)


def main(argv):
    target_dir_base = FLAGS.target_dir
    config_file = FLAGS.config
    p = pathlib2.Path(FLAGS.rosbag_dir)
    for f in p.glob('*/rosbag/*bag'):
        f_lsit = f.as_posix().split('/')
        bag_count = f_lsit[-1][:3]
        variation = f_lsit[-3]
        target_dir = os.path.join(target_dir_base, variation, bag_count)
        convert_one_rosbag(f.as_posix(), target_dir, config_file)

if __name__ == '__main__':
    app.run(main)


