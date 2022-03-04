# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import torchvision

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def gather_mpii_cooking_2_info(sequence_dir, detections, feature_dim, debug_frames=None):
    image_dir = sequence_dir
    image_filenames = sorted(os.listdir(image_dir))
    if debug_frames is not None:
        image_filenames = image_filenames[:debug_frames]
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in image_filenames}

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": None,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": 1000 / 30
    }
    return seq_info


def create_detections(detection_mat, frame_idx, feature_dim, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, track_class = row[2:6], row[6], row[7]
        if feature_dim == 256:
            track_class_logits = row[11:103]
            feature = row[103:]
        elif feature_dim == 2048:
            track_class_logits = row[11:1612]
            feature = row[1612:]
        else:
            raise ValueError('Unknown feature dim to handle')
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, track_class, track_class_logits=track_class_logits))
    return detection_list


def amend_bb_classes(bbs_classes, vg_mutual_ids):
    for i, bb_cls in enumerate(bbs_classes[:, 0]):
        if bb_cls not in vg_mutual_ids:
            bbs_classes[i, 0] = -1
    return bbs_classes


def gather_mpii_cooking_2_detections(detection_dir, seq_name, debug_frames=None, vg_mutual_classes_filepath=None):
    cls_logits_dir = os.path.join(detection_dir, 'classes_logits', seq_name)
    pred_classes_dir = os.path.join(detection_dir, 'pred_classes', seq_name)
    scores_dir = os.path.join(detection_dir, 'scores', seq_name)
    bbs_dir = os.path.join(detection_dir, 'bounding_boxes', seq_name)
    feats_dir = os.path.join(detection_dir, 'features', seq_name)
    npy_filenames = sorted(os.listdir(bbs_dir))
    if debug_frames is not None:
        npy_filenames = npy_filenames[:debug_frames]
    vg_mutual_ids = None
    if vg_mutual_classes_filepath is not None:
        vg_info = pd.read_csv(vg_mutual_classes_filepath)
        vg_ids, vg_is_mutual = vg_info['id'].tolist(), vg_info['is_mutual'].tolist()
        vg_mutual_ids = {vg_id for vg_id, vg_mutual_flag in zip(vg_ids, vg_is_mutual) if vg_mutual_flag}
    detections = []
    for i, npy_filename in enumerate(npy_filenames, start=1):
        pred_classes_filepath = os.path.join(pred_classes_dir, npy_filename)
        bbs_classes = np.expand_dims(np.load(pred_classes_filepath), axis=-1)
        if vg_mutual_ids is not None:
            bbs_classes = amend_bb_classes(bbs_classes, vg_mutual_ids)
        cls_logits_filepath = os.path.join(cls_logits_dir, npy_filename)
        cls_logits = np.load(cls_logits_filepath)
        scores_filepath = os.path.join(scores_dir, npy_filename)
        bbs_scores = np.expand_dims(np.load(scores_filepath), axis=-1)
        bbs_filepath = os.path.join(bbs_dir, npy_filename)
        bbs_tlbr = np.load(bbs_filepath)
        bbs_width, bbs_height = np.abs(bbs_tlbr[:, 2:3] - bbs_tlbr[:, 0:1]), np.abs(bbs_tlbr[:, 3:4] - bbs_tlbr[:, 1:2])
        bbs_tlwh = np.concatenate([bbs_tlbr[:, :2], bbs_width, bbs_height], axis=-1)
        frame_indices = np.full_like(bbs_scores, fill_value=i)
        first_padding_cols = np.full_like(frame_indices, fill_value=-1)
        second_padding_cols = np.full_like(bbs_tlwh[:, :3], fill_value=-1)
        mot16_cols = np.concatenate([frame_indices,  # 1  0:1
                                     first_padding_cols,  # 1  1:2
                                     bbs_tlwh,  # 4  2:6
                                     bbs_scores,  # 1  6:7
                                     bbs_classes,  # 1  7:8
                                     second_padding_cols,  # 3  8:11
                                     cls_logits,  # 1601  11:1612
                                     ],
                                    axis=-1)
        roi_features_filepath = os.path.join(feats_dir, npy_filename)
        roi_features = np.load(roi_features_filepath)
        frame_detections = np.concatenate([mot16_cols, roi_features], axis=-1)
        detections.append(frame_detections)
    detections = np.concatenate(detections, axis=0)
    return detections, 2048


def gather_detr_detections(detection_dir, seq_name, debug_frames=None):
    cls_logits_dir = os.path.join(detection_dir, 'classes_logits', seq_name)
    bbs_dir = os.path.join(detection_dir, 'bounding_boxes', seq_name)
    feats_dir = os.path.join(detection_dir, 'features', seq_name)
    npy_filenames = sorted(os.listdir(bbs_dir))
    if debug_frames is not None:
        npy_filenames = npy_filenames[:debug_frames]
    detections = []
    for i, npy_filename in enumerate(npy_filenames, start=1):
        bbs_filepath = os.path.join(bbs_dir, npy_filename)
        bbs_tlbr = np.load(bbs_filepath)
        bbs_width, bbs_height = np.abs(bbs_tlbr[:, 2:3] - bbs_tlbr[:, 0:1]), np.abs(bbs_tlbr[:, 3:4] - bbs_tlbr[:, 1:2])
        bbs_tlwh = np.concatenate([bbs_tlbr[:, :2], bbs_width, bbs_height], axis=-1)
        cls_logits_filepath = os.path.join(cls_logits_dir, npy_filename)
        cls_logits = np.load(cls_logits_filepath)
        cls_probs = softmax(cls_logits, axis=-1)[:, :-1]  # remove background class after softmax
        bbs_scores = np.max(cls_probs, axis=-1, keepdims=True)
        bbs_classes = np.reshape(np.argmax(cls_probs, axis=-1), [-1, 1])
        frame_indices = np.full_like(bbs_scores, fill_value=i)
        first_padding_cols = np.full_like(frame_indices, fill_value=-1)
        second_padding_cols = np.full_like(bbs_tlwh[:, :3], fill_value=-1)
        mot16_cols = np.concatenate([frame_indices,  # 1  0:1
                                     first_padding_cols,  # 1  1:2
                                     bbs_tlwh,  # 4  2:6
                                     bbs_scores,  # 1  6:7
                                     bbs_classes,  # 1  7:8
                                     second_padding_cols,  # 3  8:11
                                     cls_logits,  # 92  11:103
                                     ],
                                    axis=-1)
        transformer_features_filepath = os.path.join(feats_dir, npy_filename)
        transformer_features = np.load(transformer_features_filepath)
        frame_detections = np.concatenate([mot16_cols, transformer_features], axis=-1)
        detections.append(frame_detections)
    detections = np.concatenate(detections, axis=0)
    return detections, 256


def gather_bua152_detections(detection_file, seq_name, debug_frames=None):
    detections_dir = os.path.join(detection_file, seq_name)
    npz_filenames = sorted(os.listdir(detections_dir))
    if debug_frames is not None:
        npz_filenames = npz_filenames[:debug_frames]
    detections = []
    for i, npz_filename in enumerate(npz_filenames, start=1):
        frame_data = np.load(os.path.join(detections_dir, npz_filename))
        bbs_tlbr = frame_data['bounding_boxes']
        bbs_width, bbs_height = np.abs(bbs_tlbr[:, 2:3] - bbs_tlbr[:, 0:1]), np.abs(bbs_tlbr[:, 3:4] - bbs_tlbr[:, 1:2])
        bbs_tlwh = np.concatenate([bbs_tlbr[:, :2], bbs_width, bbs_height], axis=-1)
        cls_logits = frame_data['classes_logits']
        bbs_classes = np.expand_dims(frame_data['pred_classes'], axis=-1)
        bbs_scores = np.expand_dims(frame_data['scores'], axis=-1)
        # Frame + Paddings
        frame_indices = np.full_like(bbs_scores, fill_value=i)
        first_padding_cols = np.full_like(frame_indices, fill_value=-1)
        second_padding_cols = np.full_like(bbs_tlwh[:, :3], fill_value=-1)
        # Aggregate information
        mot16_cols = np.concatenate([frame_indices,  # 1  0:1
                                     first_padding_cols,  # 1  1:2
                                     bbs_tlwh,  # 4  2:6
                                     bbs_scores,  # 1  6:7
                                     bbs_classes,  # 1  7:8
                                     second_padding_cols,  # 3  8:11
                                     cls_logits,  # 1601  11:1612
                                     ],
                                    axis=-1)
        # RoI Pooled features
        roi_feats = frame_data['features']
        frame_detections = np.concatenate([mot16_cols, roi_feats], axis=-1)
        detections.append(frame_detections)
    detections = np.concatenate(detections, axis=0)
    return detections, 2048


def torchvision_nms(boxes, scores, nms_max_overlap, classes=None):
    x1 = boxes[:, 0:1]
    y1 = boxes[:, 1:2]
    x2 = boxes[:, 2:3] + boxes[:, 0:1]
    y2 = boxes[:, 3:4] + boxes[:, 1:2]
    boxes = torch.from_numpy(np.concatenate([x1, y1, x2, y2], axis=-1))
    if classes is None:
        indices = torchvision.ops.nms(boxes, torch.from_numpy(scores), nms_max_overlap)
    else:
        max_coordinate = boxes.max()
        classes = torch.from_numpy(classes)
        offsets = classes.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        indices = torchvision.ops.nms(boxes_for_nms, torch.from_numpy(scores), nms_max_overlap)
    return indices.tolist()


def run(sequence_dir, detection_file, output_dir, min_confidence, nms_strategy,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, debug_frames, save_tracklets_features, vg_mutual_classes_filepath=None):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to directory containing video frames.
    detection_file : str
        Path to directory containing object detection of all videos in subdirectories.
    output_dir : str
        Path to the root directory to save output tracking. A subdirectory is created depending on object detection
        configuration and the current tracking configuration, and a .txt file with the video tracking is saved inside
        it.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_strategy: str
        Which NMS library algorithm to apply. Options are: standard, torchvision, or torchvision-class.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    debug_frames: Optional[int]
        If not None, only tracks the first specified number of frames.
    save_tracklets_features: bool
        Whether to save extra information, such as visual features, other than the bounding box location and id.
    vg_mutual_classes_filepath: Optional[str]
        If specified and the detections come from a model trained on Visual Genome, map unknown classes to a single
        unknown class.
    """
    detection_cfg = os.path.basename(detection_file)
    nms_log = {'standard': 's', 'torchvision': 't', 'torchvision-class': 'tc'}[nms_strategy]
    tracking_cfg = (str(min_confidence) +
                    '-' + nms_log +
                    '-' + str(nms_max_overlap) +
                    '-' + str(int(save_tracklets_features))
                    )
    save_subdir = detection_cfg + '_' + tracking_cfg
    save_dir = os.path.join(output_dir, save_subdir)
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    seq_name = os.path.basename(sequence_dir)
    save_filename = os.path.join(save_dir, seq_name + '.txt')
    if os.path.isfile(save_filename):
        print('Tracking for video %s already done. Skipping it.' % seq_name)
        return
    # detection_file is actually a dir below
    if 'detr' in detection_file:  # detections from detr
        collated_detections, feature_dim = gather_detr_detections(detection_file, seq_name, debug_frames=debug_frames)
    elif 'pbua' in detection_file:  # detections from py-bottom-up-attention
        collated_detections, feature_dim = \
            gather_mpii_cooking_2_detections(detection_file, seq_name,
                                             debug_frames=debug_frames,
                                             vg_mutual_classes_filepath=vg_mutual_classes_filepath)
    elif 'bua' in detection_file:  # detections from bottom-up-attention.pytorch
        collated_detections, feature_dim = gather_bua152_detections(detection_file, seq_name, debug_frames=debug_frames)
    else:
        raise ValueError('Unknown detections to track.')
    seq_info = gather_mpii_cooking_2_info(sequence_dir, collated_detections, feature_dim, debug_frames=debug_frames)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, feature_dim, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if nms_strategy == 'standard':
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
        elif nms_strategy == 'torchvision':
            indices = torchvision_nms(boxes, scores, nms_max_overlap)
        else:
            classes = np.array([d.track_class for d in detections])
            indices = torchvision_nms(boxes, scores, nms_max_overlap, classes)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                               frame_idx, track.track_id, track.track_class, track.confidence,
                               bbox[0], bbox[1], bbox[2], bbox[3]
                           ] +
                           track.track_class_logits.tolist() +
                           track.latest_feature.tolist()
                           )

    # Run tracker
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results
    f = open(save_filename, 'w')
    for row in results:
        first_str_data = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        first_str = '%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % first_str_data
        final_str = first_str
        if save_tracklets_features:
            second_str_data = tuple(row[8:])
            second_str = ','.join(['%.6f'] * len(second_str_data))
            second_str = second_str % second_str_data
            final_str = first_str + "," + second_str
        print(final_str, file=f)


def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return input_string == "True"


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to directory containing video frames.",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to directory containing object detection of all videos in subdirectories.",
        default=None, required=True)
    parser.add_argument(
        "--output_dir", help="Path to the tracking output dir. A sub-directory will be created to write a file "
                             "containing the tracking results on completion.",
        default="/tmp")
    parser.add_argument(
        '--vg_mutual_classes_filepath', type=str,
        help='CSV file containing the mapping between VG classes and MPII/EK vocabulary.'
    )
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        '--nms_strategy', default='standard',
        help='One of: standard, torchvision, or torchvision-class.')
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--debug_frames",
        help="Number of initial frames from a video to check detection and tracking. If None, track "
             "the whole video.",
        type=int, default=None)
    parser.add_argument(
        "--save_tracklets_features",
        help="Whether to save extra information other than bounding box location and id.",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        sequence_dir=args.sequence_dir,
        detection_file=args.detection_file,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        nms_strategy=args.nms_strategy,
        nms_max_overlap=args.nms_max_overlap,
        min_detection_height=args.min_detection_height,
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=args.nn_budget,
        display=args.display,
        debug_frames=args.debug_frames,
        save_tracklets_features=args.save_tracklets_features,
        vg_mutual_classes_filepath=args.vg_mutual_classes_filepath)
