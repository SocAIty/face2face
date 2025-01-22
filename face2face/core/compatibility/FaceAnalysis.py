# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime

__all__ = ['FaceAnalysis']

from insightface.model_zoo import ArcFaceONNX

from face2face.core.compatibility.Attribute import Attribute
from face2face.core.compatibility.Face import Face
from face2face.core.compatibility.Landmark import Landmark
from face2face.core.compatibility.retinaface import RetinaFace


class FaceAnalysis:
    def __init__(self, model_dir: str, session):

        onnxruntime.set_default_logger_severity(3)

        # loading all the insightface buffalo models and storing them in a dictionary
        self.models = {}
        onnx_files = glob.glob(osp.join(model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = self._load_model(onnx_file, session)
            self.models[model.taskname] = model

        self.det_model = self.models['detection']

    def _load_model(self, onnx_file: str, session):
        if "1k3d68" in onnx_file or "2d106det" in onnx_file:
            return Landmark(model_file=onnx_file, session=session)
        if "det_10g" in onnx_file:
            return RetinaFace(model_file=onnx_file, session=session)
        if "genderage" in onnx_file:
            return Attribute(model_file=onnx_file, session=session)
        if "w600k_r50" in onnx_file:
            return ArcFaceONNX(model_file=onnx_file, session=session)

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg

