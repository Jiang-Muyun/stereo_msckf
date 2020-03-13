import sys
import time
import argparse
import cv2
from queue import Queue
from threading import Thread

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF

from dataset import EuRoCDataset, DataPublisher


class VIO(object):
    def __init__(self, config, img_queue, imu_queue):
        self.config = config
        self.img_queue = img_queue
        self.imu_queue = imu_queue

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)

        self.img_thread.start()
        self.imu_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                return

            cv2.imshow('cam0', img_msg.cam0_image)
            cv2.waitKey(1)

            feature_msg = self.image_processor.stareo_callback(img_msg)
            self.msckf.feature_callback(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/media/james/MX_NTFS/dataset/EuRoc/MH_01_easy', help='Path of EuRoC')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    args = parser.parse_args()

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)   # start from static state

    img_queue = Queue()
    imu_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue)

    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)