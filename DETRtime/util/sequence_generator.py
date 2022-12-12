import numpy as np
from util.box_ops import box_cxw_to_xlxh, box_xlxh_to_cxw
import torch

def boxes_to_cp(boxes):
    cx, w = boxes[:, 0], boxes[:, 1]
    x_low = cx - 0.5 * w
    return torch.stack([x_low], dim=-1)


def generate_sequence_targets(targets, timestamps):
    '''
    In:
        targets: {
            'boxes': list of lists with [[center, width], ... ], (shape: (#boxes, 2))
            'labels': list of classes i.e. [0, 1, 1, ...] (shape: (#boxes, ))
        }
    Out:
        np.array of shape (sample_width, )
    '''

    seq = np.zeros(timestamps)

    boxes = boxes_to_cp(targets['boxes'])
    if len(boxes) == 0:
        return seq


    for i in range(len(boxes) - 1):
        label = targets['labels'][i]
        left = int(boxes[i][0] * timestamps)
        right = int(boxes[i+1][0] * timestamps)

        left = max(0, left)
        right = min(timestamps, right)
        event = label.cpu().detach().numpy()
        # assert right <= timestamps
        # assert left >= 0
        # assert right >= left

        seq[left:right] = np.ones(right-left) * event
            
    label = targets['labels'][-1]
    event = label.cpu().detach().numpy()
    left = max(0, int(boxes[-1][0] * timestamps))
    seq[left:timestamps] = np.ones(timestamps-left) * event

    return seq

def generate_sequence_predictions(predictions, timestamps):
    '''
    In:
        predictions: {
            'pred_boxes': list of lists with [[center, width], ... ], (shape: (N, 2))
            'pred_logits': list of classes i.e. [[c_1, c_2, ...], [c_1, c_2, ...], ...] (shape: (N, #classes))
        }
    Out:
        np.array of shape (sample_width, )
    '''

    seq = np.zeros(timestamps)
    boxes = boxes_to_cp(predictions['pred_boxes'])
    combined = [x for x in zip(boxes, predictions['pred_logits'])] # [([x,w],[c1,...,c4]), ...]
    combined.sort(key=lambda obj: obj[0][0])

    left = 0
    label = torch.Tensor([0])
    event = label.cpu().detach().numpy()

    for i in range(len(combined)):

        while (i < len(combined) and np.argmax(combined[i][1]) == len(combined[i][1]) - 1):
            i += 1

        if (i == len(combined)):
            break

        right = min(timestamps, max(left, int(combined[i][0][0] * timestamps)))
        seq[left:right] = np.ones(right-left) * event

        label = np.argmax(combined[i][1])
        event = label.cpu().detach().numpy()
        left = max(0, right)

        assert right <= timestamps
        assert left >= 0
        assert right >= left 
                       
    # last handcrafted
    seq[left:timestamps] = np.ones(timestamps-left) * event

    return seq

# def generate_sequence_targets(targets, timestamps):
#     '''
#     In:
#         targets: {
#             'boxes': list of lists with [[center, width], ... ], (shape: (#boxes, 2))
#             'labels': list of classes i.e. [0, 1, 1, ...] (shape: (#boxes, ))
#         }
#     Out:
#         np.array of shape (sample_width, )
#     '''

#     seq = np.zeros(timestamps)

#     boxes = targets['boxes']
#     if len(boxes) > 0:
#         boxes = box_cxw_to_xlxh(boxes)
#     else:
#         return seq

#     for i in range(len(boxes)):
#         box = boxes[i]
#         label = targets['labels'][i]
#         left = int(box[0] * timestamps)
#         right = int(box[1] * timestamps)
#         left = max(0, int(box[0] * timestamps))
#         right = min(timestamps, int(box[1] * timestamps))
#         event = label.cpu().detach().numpy()
#         # assert right <= timestamps
#         # assert left >= 0
#         # assert right >= left

#         seq[left:right] = np.ones(right-left) * event

#     return seq

# def generate_sequence_predictions(predictions, timestamps):
#     '''
#     In:
#         predictions: {
#             'pred_boxes': list of lists with [[center, width], ... ], (shape: (N, 2))
#             'pred_logits': list of classes i.e. [[c_1, c_2, ...], [c_1, c_2, ...], ...] (shape: (N, #classes))
#         }
#     Out:
#         np.array of shape (sample_width, )
#     '''

#     seq = np.zeros(timestamps)
#     boxes = predictions['pred_boxes']
#     boxes = box_cxw_to_xlxh(boxes)
#     combined = [x for x in zip(boxes, predictions['pred_logits'])] # [([x,w],[c1,...,c4]), ...]
#     combined.sort(key=lambda obj: max(obj[1]))

#     for i in range(len(combined)):
#         label = np.argmax(combined[i][1])

#         if label == len(combined[i][1]) - 1:
#             continue

#         box = combined[i][0]
#         # event = 0 if label == 0 else 2 # for 2 class
#         event = label.cpu().detach().numpy()

#         left = max(0, int(box[0] * timestamps))
#         right = min(timestamps, int(box[1] * timestamps))
        
#         assert right <= timestamps
#         assert left >= 0
#         assert right >= left

#         seq[left:right] = np.ones(right-left) * event

#     return seq