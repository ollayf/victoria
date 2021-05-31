import torch
import cv2
import time
from scipy import stats
import numpy as np
import getpass

from Tracker import ModeCounter
from Stream import ProcessedStream
from openpifpaf import transforms

class Master:

    # basic preprocess
    preprocess = transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.CenterPadTight(16),
        transforms.EVAL_TRANSFORM,
    ])

    def __init__(self, model, processor, tracker=None, stream=None):
        self.model = model
        self.processor = processor
        self.stream = self._prepare_stream(stream)
        self.tracker = tracker(self)

    def _prepare_stream(self, stream):
        '''
        Allows taking in direct instance of ProcessedStream
        Allows rtsp
        '''
        if not stream:
            return ProcessedStream(0, preprocess=self.preprocess)
        if isinstance(stream, ProcessedStream):
            return stream
        return ProcessedStream(stream, preprocess=self.preprocess)
        
    def start_detection(self):
        self.start_time = time.perf_counter()
        loop = self.start_time
        for frame, processed_image, _, meta in self.stream:
            preds = self.processor.batch(self.model, torch.unsqueeze(processed_image, 0), device='cuda:0')[0]

            if preds:
                for ann in preds:
                    ann = ann.inverse_transform(meta)
                    json_data = ann.json_data()
                    frame.add_prediction(json_data)

            if frame.image is not None:
                cv2.imshow('Test', frame.annotated())
                cv2.waitKey(1)
            
            # for superficial tracking
            person_count = len(frame)
            self.tracker.update(person_count)

            # print out fps statistics
            print("FPS:", 1/(time.perf_counter()-loop))
            loop = time.perf_counter()