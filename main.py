from Tracker import ModeCounter
from Master import Master
import os
import glob
import json
import copy
import logging
import time
from statistics import mode
from scipy import stats
import numpy as np

import cv2
import torch
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show, logger, transforms

from utils import annotation, mutils
# OPENPIFPAF_MODEL = 'https://drive.google.com/uc?id=1b408ockhh29OLAED8Tysd2yGZOo0N_SQ'
# package_model_url = r'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12.6/shufflenetv2k16-210404-110105-cocokp-o10s-f90ed364.pkl'

# processor, pifpaf_model = processor_factory(args)
# preprocess = preprocess_factory(args)

# for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
#     pred_batch = processor.batch(pifpaf_model, image_tensors_batch, device=args.device)

#     # unbatch (only for MonStereo)
#     for idx, (pred, meta) in enumerate(zip(pred_batch, meta_batch)):
#         LOG.info('batch %d: %s', batch_i, meta['file_name'])
#         pred = [ann.inverse_transform(meta) for ann in pred]

# checkpoint = torch.hub.load_state_dict_from_url(
#     package_model_url,
#     check_hash=True,
#     progress=True
# )

VIDEO = '/home/hosea/Infinergy/videos/sieved/busy/busy.mp4'

downloaded_ckpt = 'models/pifpaf/shufflenetv2k16-210404-110105-cocokp-o10s-f90ed364.pkl'
model, processor = mutils.get_model(downloaded_ckpt)

# print(checkpoint)
# print(type (checkpoint))
# print()
# print()
# for key in checkpoint.keys():
#     print(key)

master = Master(model, processor, ModeCounter, VIDEO)

master.start_detection()