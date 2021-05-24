import os
import glob
import json
import copy
import logging
import time
from collections import defaultdict
from typing_extensions import final

import cv2
import torch
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf.stream import Stream
from openpifpaf.predict import processor_factory, preprocess_factory
from openpifpaf import decoder, network, visualizer, show, logger, transforms

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

checkpoint = torch.load(r'models/pifpaf/shufflenetv2k16-210404-110105-cocokp-o10s-f90ed364.pkl')

# print(checkpoint)
# print(type (checkpoint))
# print()
# print()
# for key in checkpoint.keys():
#     print(key)


preprocess = transforms.Compose([
        transforms.NormalizeAnnotations(),
        transforms.CenterPadTight(16),
        transforms.EVAL_TRANSFORM,
    ])

# stream = Stream('/home/hosea/Infinergy/videos/sieved/09.13.19-09.13.55.mp4', preprocess=preprocess)
stream = Stream(2, preprocess=preprocess)

net_cpu = checkpoint['model']
print(net_cpu)
print(type(net_cpu))
# initialise for eval
net_cpu.eval()
model = net_cpu.to('cuda:0')
processor = decoder.factory(net_cpu.head_metas)

def annotate_image(im_array, coords):
    final_img = im_array.copy()
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    for x, y in coords:
        final_img = cv2.circle(final_img, (x, y), 5, [0, 0, 255], -1)
        
    cv2.imshow('Test', final_img)
    cv2.waitKey(1)


def parse_pred_json(json, value):
    print(json)
    kp = json['keypoints']
    print('Number of keypoints is', len(kp)/3)
    x, y = kp[value*3], kp[value*3+1]
    coords = [(int(x), int(y)),]
    return coords

def annotate_box(im_array, json):
    bbox = json['bbox']
    print(bbox)
    x1, y1, w, h = list(map(lambda x: int(x), bbox))
    x2 = x1 + w
    y2 = y1 + h
    score = json['score']
    final_img = im_array.copy()
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    final_img = cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    final_img = cv2.putText(final_img, "Score: {}".format(score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0 , 255))
    cv2.imshow('BBox', final_img)
    cv2.waitKey(1)


start_loop = time.perf_counter()
loop = start_loop
for image, processed_image, _, meta in stream:
    preds = processor.batch(model, torch.unsqueeze(processed_image, 0), device='cuda:0')[0]

    input()
    print(preds)
    print('Preds type:', type(preds))
    if preds:
        preds = [ann.inverse_transform(meta) for ann in preds]
        # coords = parse_pred_json(preds[0].json_data(), 2)
        annotate_box(image, preds[0].json_data())
    else:
        if image:
            cv2.imshow('Test', processed_image)
            cv2.waitKey(1)

    # print out fps statistics
    print("FPS:", 1/(time.perf_counter()-loop))
    loop = time.perf_counter()