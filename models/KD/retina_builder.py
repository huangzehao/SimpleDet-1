from __future__ import division
from __future__ import print_function

import math
import mxnet as mx
import mxnext as X

from symbol.builder import RpnHead, Backbone, Neck
from models.retinanet.builder import RetinaNetHead


class RetinaNet(object):
    def __init__(self):
        pass

    @staticmethod
    def get_train_symbol(backbone, neck, head, FitParam):
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")
        teacher_label = X.var("teacher_label")

        feat = backbone.get_rpn_feature()
        c2, c3, c4, c5 = feat
        feat_dict = {'c2': c2,
                     'c3': c3,
                     'c4': c4,
                     'c5': c5}
        feat = neck.get_rpn_feature(feat)

        loss = head.get_loss(feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)

        mimic_feat = feat_dict[FitParam.mimic_stage]
        mimic_channel = FitParam.mimic_channel
        mimic_grad_scale = FitParam.mimic_grad_scale

        student_hint = mx.sym.Convolution(data=mimic_feat, num_filter=mimic_channel, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name="student_hint_conv")
        student_hint = mx.sym.Activation(data=student_hint, act_type='relu', name="student_hint_relu")
        fit_loss = mx.sym.mean(mx.sym.square(student_hint - teacher_label))
        fit_loss = mx.sym.MakeLoss(fit_loss, grad_scale=mimic_grad_scale)

        return X.group(loss + (fit_loss, ))

    @staticmethod
    def get_test_symbol(backbone, neck, head):
        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        feat = backbone.get_rpn_feature()
        feat = neck.get_rpn_feature(feat)

        cls_score, bbox_xyxy = head.get_prediction(feat, im_info)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])


