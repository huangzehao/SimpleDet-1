import mxnet as mx
import mxnext as X
from symbol.builder import BboxHead


class KLBboxHead(BboxHead):
    def __init__(self, pBbox):
        super(KLBboxHead, self).__init__(pBbox)

    def get_output(self, conv_feat):
        p = self.p
        num_class = p.num_class
        num_reg_class = 2 if p.regress_target.class_agnostic else num_class

        head_feat = self._get_bbox_head_logit(conv_feat)

        if p.fp16:
            head_feat = X.to_fp32(head_feat, name="bbox_head_to_fp32")

        cls_logit = X.fc(
            head_feat,
            filter=num_class,
            name='bbox_cls_logit',
            init=X.gauss(0.01)
        )
        bbox_delta = X.fc(
            head_feat,
            filter=4 * num_reg_class,
            name='bbox_reg_delta',
            init=X.gauss(0.001)
        )

        bbox_var = X.fc(
            head_feat,
            filter=4 * num_reg_class,
            name="bbox_var",
            init=X.gauss(0.0001)
        )

        return cls_logit, bbox_delta, bbox_var

    def get_prediction(self, conv_feat, im_info, proposal):
        p = self.p
        bbox_mean = p.regress_target.mean
        bbox_std = p.regress_target.std
        batch_image = p.batch_image
        num_class = p.num_class
        class_agnostic = p.regress_target.class_agnostic
        num_reg_class = 2 if class_agnostic else num_class

        cls_logit, bbox_delta, bbox_var = self.get_output(conv_feat)

        bbox_delta = X.reshape(
            bbox_delta,
            shape=(batch_image, -1, 4 * num_reg_class),
            name='bbox_delta_reshape'
        )

        bbox_xyxy = X.decode_bbox(
            rois=proposal,
            bbox_pred=bbox_delta,
            im_info=im_info,
            name='decode_bbox',
            bbox_mean=bbox_mean,
            bbox_std=bbox_std,
            class_agnostic=class_agnostic
        )
        cls_score = X.softmax(
            cls_logit,
            axis=-1,
            name='bbox_cls_score'
        )
        cls_score = X.reshape(
            cls_score,
            shape=(batch_image, -1, num_class),
            name='bbox_cls_score_reshape'
        )
        return cls_score, bbox_xyxy

    def get_loss(self, conv_feat, cls_label, bbox_target, bbox_weight):
        p = self.p
        batch_roi = p.image_roi * p.batch_image
        batch_image = p.batch_image

        cls_logit, bbox_delta, bbox_var = self.get_output(conv_feat)

        bbox_var_exp = mx.sym.exp(-bbox_var)

        scale_loss_shift = 128.0 if p.fp16 else 1.0

        # classification loss
        cls_loss = X.softmax_output(
            data=cls_logit,
            label=cls_label,
            normalization='batch',
            grad_scale=1.0 * scale_loss_shift,
            name='bbox_cls_loss'
        )

        # bounding box regression
        reg_loss = X.smooth_l1(
            bbox_delta - bbox_target,
            scalar=1.0,
            name='bbox_reg_l1'
        )
        reg_loss = bbox_weight * reg_loss
        if p.reg_loss_scale_alpha:
            reg_loss = reg_loss * bbox_var_exp

        reg_loss = X.loss(
            reg_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='bbox_reg_loss',
        )

        # bbox var loss
        bbox_delta_copy = mx.sym.identity(bbox_delta, name="bbox_delta_copy")
        bbox_delta_copy = X.block_grad(bbox_delta_copy, name="bbox_delta_copy_blockgrad")
        reg_loss_copy = X.smooth_l1(
            bbox_delta_copy - bbox_target,
            scalar=1.0,
            name='bbox_reg_l1_copy'
        )
        bbox_var_loss = (bbox_var_exp * reg_loss_copy) + (bbox_var / 2.0)
        bbox_var_loss = bbox_weight * bbox_var_loss
        bbox_var_loss = X.loss(
            bbox_var_loss,
            grad_scale=1.0 / batch_roi * scale_loss_shift,
            name='bbox_var_loss'
        )

        # append label
        cls_label = X.reshape(
            cls_label,
            shape=(batch_image, -1),
            name='bbox_label_reshape'
        )
        cls_label = X.block_grad(cls_label, name='bbox_label_blockgrad')

        # output
        return cls_loss, reg_loss, bbox_var_loss, cls_label


class KLFPNBbox2fcHead(KLBboxHead):
    def __init__(self, pBbox):
        super(KLFPNBbox2fcHead, self).__init__(pBbox)

    def _get_bbox_head_logit(self, conv_feat):
        if self._head_feat is not None:
            return self._head_feat

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        flatten = X.flatten(conv_feat, name="bbox_feat_flatten")
        fc1 = X.fc(flatten, filter=1024, name="bbox_fc1", init=xavier_init)
        fc1 = X.relu(fc1)
        fc2 = X.fc(fc1, filter=1024, name="bbox_fc2", init=xavier_init)
        fc2 = X.relu(fc2)

        self._head_feat = fc2

        return self._head_feat