import mmcv
import numpy as np
from .visualiser import Visualiser


EPS = 1e-2


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      score_thr=0.3,
                      show=True,
                      out_file=None):
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # img = mmcv.bgr2rgb(img)
    img = np.ascontiguousarray(img)
    visualiser = Visualiser(dataset="VID")

    visualiser.add_img(img, img_id="out_pred")
    for bbox, label in zip(bboxes, labels):
        visualiser.add_coco_bbox(
            bbox[:4], label+1, bbox[4], img_id="out_pred")

    # visualiser.add_img(img, img_id="out_gt")
    # for k in range(len(dets_gt[i])):
    #     debugger.add_coco_bbox(
    #         dets_gt[i, k, :4], labels_gt[k], 1.0, img_id="out_gt"
    #     )

    if show:
        visualiser.show_all_imgs(pause=True)
    if out_file is not None:
        mmcv.imwrite(visualiser.get_all_imgs(), out_file)

    return img
