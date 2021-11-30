# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
from scipy.special import softmax

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


def gather_mpii_cooking_2_info(sequence_dir, detections, debug_frames=None):
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

    feature_dim = detections.shape[1] - 11 if detections is not None else 0
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


def create_detections(detection_mat, frame_idx, min_height=0):
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
        bbox, confidence, track_class, feature = row[2:6], row[6], row[7], row[11:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, track_class))
    return detection_list


def gather_mpii_cooking_2_detections(detection_dir, seq_name, debug_frames=None):
    bbs_dir = os.path.join(detection_dir, 'bounding_boxes', seq_name)
    feats_dir = os.path.join(detection_dir, 'roi_features', seq_name)
    npy_filenames = sorted(os.listdir(bbs_dir))
    if debug_frames is not None:
        npy_filenames = npy_filenames[:debug_frames]
    detections = []
    for i, npy_filename in enumerate(npy_filenames, start=1):
        bbs_filepath = os.path.join(bbs_dir, npy_filename)
        bbs = np.load(bbs_filepath)
        try:
            bbs_tlbr = bbs['bbox']
        except IndexError:
            bbs_tlbr = bbs[:, :4]
        try:
            bbs_scores = bbs['scores']  # (num_bboxes, num_classes + 1)
            bbs_classes = np.argmax(bbs_scores[:, 1:], axis=-1) + 1
            bbs_classes = np.reshape(bbs_classes, (-1, 1))
            bbs_scores = np.max(bbs_scores[:, 1:], axis=-1, keepdims=True)
        except KeyError:
            bbs_scores = np.full_like(bbs_tlbr[:, :1], fill_value=1)
            bbs_classes = np.full_like(bbs_scores, fill_value=-1)
        except IndexError:
            bbs_scores, bbs_classes = bbs[:, 4:5], bbs[:, 5:]
        bbs_width, bbs_height = np.abs(bbs_tlbr[:, 2:3] - bbs_tlbr[:, 0:1]), np.abs(bbs_tlbr[:, 3:4] - bbs_tlbr[:, 1:2])
        bbs_tlwh = np.concatenate([bbs_tlbr[:, :2], bbs_width, bbs_height], axis=-1)
        frame_indices = np.full_like(bbs_scores, fill_value=i)
        first_padding_cols = np.full_like(frame_indices, fill_value=-1)
        second_padding_cols = np.full_like(bbs_tlwh[:, :3], fill_value=-1)
        mot16_cols = np.concatenate([frame_indices, first_padding_cols, bbs_tlwh, bbs_scores,
                                     bbs_classes, second_padding_cols],
                                    axis=-1)
        roi_features_filepath = os.path.join(feats_dir, npy_filename)
        roi_features = np.load(roi_features_filepath)
        try:
            roi_features = roi_features['x']
        except IndexError:
            pass
        frame_detections = np.concatenate([mot16_cols, roi_features], axis=-1)
        detections.append(frame_detections)
    detections = np.concatenate(detections, axis=0)
    return detections


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
        mot16_cols = np.concatenate([frame_indices, first_padding_cols, bbs_tlwh, bbs_scores,
                                     bbs_classes, second_padding_cols],
                                    axis=-1)
        transformer_features_filepath = os.path.join(feats_dir, npy_filename)
        transformer_features = np.load(transformer_features_filepath)
        frame_detections = np.concatenate([mot16_cols, transformer_features], axis=-1)
        detections.append(frame_detections)
    detections = np.concatenate(detections, axis=0)
    return detections


def run(sequence_dir, detection_file, output_dir, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, debug_frames):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_dir : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
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
    """
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    seq_name = os.path.basename(sequence_dir)
    # detection_file is actually a dir below
    if 'detr' in detection_file:
        collated_detections = gather_detr_detections(detection_file, seq_name, debug_frames=debug_frames)
    else:
        collated_detections = gather_mpii_cooking_2_detections(detection_file, seq_name, debug_frames=debug_frames)
    seq_info = gather_mpii_cooking_2_info(sequence_dir, collated_detections, debug_frames=debug_frames)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
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
                frame_idx, track.track_id, track.track_class, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    detection_cfg = os.path.basename(detection_file)
    tracking_cfg = str(min_confidence) + '-' + str(nms_max_overlap)
    save_subdir = detection_cfg + '_' + tracking_cfg
    save_dir = os.path.join(output_dir, save_subdir)
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    save_filename = os.path.join(save_dir, seq_name + '.txt')
    f = open(save_filename, 'w')
    for row in results:
        print('%d,%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_dir", help="Path to the tracking output dir. A sub-directory will be created to write a file "
                             "containing the tracking results on completion.",
        default="/tmp")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
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
    parser.add_argument("--debug_frames",
                        help="Number of initial frames from a video to check detection and tracking. If None, track "
                             "the whole video.",
                        type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_dir,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.debug_frames)
