'''
Cannibalised Stream class from openpifpaf library
'''

import argparse
import logging
import time
import getpass

import numpy as np
import torch

import Frame

try:
    import cv2  # pylint: disable=import-error
except ImportError:
    cv2 = None

import PIL
try:
    import PIL.ImageGrab
except ImportError:
    pass

try:
    import mss
except ImportError:
    mss = None

LOG = logging.getLogger(__name__)


# pylint: disable=abstract-method
class ProcessedStream(torch.utils.data.IterableDataset):
    horizontal_flip = None
    rotate = None
    crop = None
    scale = 1.0
    start_frame = None
    start_msec = None
    max_frames = None

    def __init__(self, source, *,
                 preprocess=None,
                 with_raw_image=True,
                 start=0,
                 # Simple Transforms
                 horizontal_flip=None,
                 rotate=None,
                 crop=None,
                 scale=1.0,
                 **kwargs):
        '''
        @params start: the starting point of the stream
        Simple Transforms: any simple transforms. It should be a dictionary of transforms like
            'horizontal_flip': (bool),
            'rotate': degrees (int),
            'crop: (left, top, right, bottom) (ints),
            'scale': x (int or float),
        '''
        super().__init__()

        self.source = source
        self.preprocess = preprocess
        self.with_raw_image = with_raw_image
        
        self.cap = self._initialiseVideoCap(source)
        self.fps = self.get_fps()
        self._prepare_start(start)

        self.horizontal_flip = horizontal_flip
        self.rotate = rotate
        self.crop = crop
        self.scale = scale

    def _prepare_start(self, start):
        '''
        Skips the footage as per requested
        '''
        if isinstance(start, int):
            self.start_frame = start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        else:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start)

    @staticmethod
    def addAuthIntoRtsp(user, pwd, url):
        params = url.split('/')[1:]
        params = list(filter(lambda x: bool(x), params))
        url_front = 'rtsp://{user}:{pwd}@{ip_port}'.format(user=user, pwd=pwd, ip_port=params[0])
        params[0] = url_front
        url = '/'.join(params)
        return url

    def _initialiseVideoCap(self, source, max_tries=3):
        if isinstance(source, str):
            if source == 'screen':
                return None
            
            # If it is rtsp
            if source[:5].lower() == 'rtsp:':
                tries = 0


                while tries <= max_tries:
                    username = input("Username: ")
                    pwd = getpass.getpass("Password: ")
                    rtsp_url = self.addAuthIntoRtsp(username, pwd, source)
                    cap = cv2.VideoCapture(rtsp_url)
                    if cap.isOpened():
                        return cap

                raise Exception("Invalid RTSP URL/ Authentication")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception("Invalid Stream input")
        return cap

    def get_fps(self):
        '''
        Self-written for more seamless experience
        '''
        if self.source == 'screen':
            return None
        return self.cap.get(cv2.CAP_PROP_FPS)

    # pylint: disable=unsubscriptable-object
    def preprocessing(self, image):
        if self.scale != 1.0:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale)
            LOG.debug('resized image size: %s', image.shape)
        if self.horizontal_flip:
            image = image[:, ::-1]
        if self.crop:
            if self.crop[0]:
                image = image[:, self.crop[0]:]
            if self.crop[1]:
                image = image[self.crop[1]:, :]
            if self.crop[2]:
                image = image[:, :-self.crop[2]]
            if self.crop[3]:
                image = image[:-self.crop[3], :]
        if self.rotate == 'left':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=0)
        elif self.rotate == 'right':
            image = np.swapaxes(image, 0, 1)
            image = np.flip(image, axis=1)
        elif self.rotate == '180':
            image = np.flip(image, axis=0)
            image = np.flip(image, axis=1)

        image_pil = PIL.Image.fromarray(np.ascontiguousarray(image))
        meta = {
            'hflip': False,
            'offset': np.array([0.0, 0.0]),
            'scale': np.array([1.0, 1.0]),
            'valid_area': np.array([0.0, 0.0, image_pil.size[0], image_pil.size[1]]),
        }
        processed_image, anns, meta = self.preprocess(image_pil, [], meta)
        return image, processed_image, anns, meta

    # pylint: disable=too-many-branches
    def __iter__(self):
        if self.source == 'screen':
            capture = 'screen'
            if mss is None:
                print('!!!!!!!!!!! install mss (pip install mss) for faster screen grabs')

        frame_start = 0 if not self.start_frame else self.start_frame
        frame_i = frame_start
        while True:
            frame_i += 1
            if self.max_frames and frame_i - frame_start > self.max_frames:
                LOG.info('reached max frames %d', self.max_frames)
                break

            if self.cap:
                _, image = self.cap.read()
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                if mss is None:
                    image = np.asarray(PIL.ImageGrab.grab().convert('RGB'))
                else:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        image = np.asarray(sct.grab(monitor))[:, :, 2::-1]
            if image is None:
                LOG.info('no more images captured')
                break

            start_preprocess = time.perf_counter()
            image, processed_image, anns, meta = self.preprocessing(image)
            meta['frame_i'] = frame_i
            meta['preprocessing_s'] = time.perf_counter() - start_preprocess

            if self.with_raw_image:
                self.frame = Frame.Frame(image)
                yield self.frame, processed_image, anns, meta
            else:
                yield processed_image, anns, meta