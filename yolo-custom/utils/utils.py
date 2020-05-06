import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, .0, .02)
    elif classname.find('BatchNorm2D') != -1:
        nn.init.normal_(m.weight.data, 1., .02)
        nn.init.constant_(m.bias.data, .0)


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()


def rescale_boxes(boxes, current_dim, original_shape):
    orig_h, orig_w = original_shape
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by score
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match

            weights = detections[invalid, 4:5]
            # Remove overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nr_bboxes = pred_boxes.size(0)
    nr_anchors = pred_boxes.size(1)
    nr_classes = pred_cls.size(-1)
    grid_size = pred_boxes.size(2)

    obj_mask = ByteTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    noobj_mask = ByteTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(1)
    class_mask = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    iou_scores = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    target_x = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    target_y = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    target_w = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    target_h = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size).fill_(0)
    target_cls = FloatTensor(nr_bboxes, nr_anchors, grid_size, grid_size, nr_classes).fill_(0)

    target_boxes = target[:, 2:6] * grid_size  # here adjust to grid
    target_xy_on_grid = target_boxes[:, :2]
    target_wh_on_grid = target_boxes[:, 2:]
    batch_idx, target_labels = target[:, :2].long().t()
    cell_x, cell_y = target_xy_on_grid.t()
    cell_w, cell_h = target_wh_on_grid.t()
    cell_i, cell_j = target_xy_on_grid.long().t()  # integer indices of cells
    
    # find what anchor fits bbox the best
    ious = torch.stack([bbox_wh_iou(anchor, target_wh_on_grid) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    
    # set cell that is responsible for detecting the object
    obj_mask[batch_idx, best_n, cell_j, cell_i] = 1
    # set noobj to False where iou is greater than threshold
    noobj_mask[batch_idx, best_n, cell_j, cell_i] = 0
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[batch_idx[i], anchor_ious > ignore_thres, cell_j[i], cell_i[i]] = 0

    # see YOLO forward
    target_x[batch_idx, best_n, cell_j, cell_i] = cell_x - cell_x.floor()
    target_y[batch_idx, best_n, cell_j, cell_i] = cell_y - cell_y.floor()
    target_w[batch_idx, best_n, cell_j, cell_i] = torch.log(cell_w / anchors[best_n][:, 0] + 1e-16)
    target_h[batch_idx, best_n, cell_j, cell_i] = torch.log(cell_h / anchors[best_n][:, 1] + 1e-16)
    target_cls[batch_idx, best_n, cell_j, cell_i, target_labels] = 1
    
    # class and coordinate predictions
    class_mask[batch_idx, best_n, cell_j, cell_i] = (pred_cls[batch_idx, best_n, cell_j, cell_i].argmax(-1) == target_labels).float()
    iou_scores[batch_idx, best_n, cell_j, cell_i] = bbox_iou(pred_boxes[batch_idx, best_n, cell_j, cell_i], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, target_x, target_y, target_w, target_h, target_cls, tconf
