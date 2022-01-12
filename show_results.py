# vim: expandtab:ts=4:sw=4
import argparse

import cv2
import numpy as np

import deep_sort_app
from deep_sort.iou_matching import iou
from application_util import postprocessing
from application_util import visualization


COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
DEFAULT_UPDATE_MS = 20


def create_id_to_name_dict(filepath):
    d = {}
    with open(filepath, mode='r') as f:
        for i, line in enumerate(f, start=0):
            obj_name = line.split(',')[0].lower().strip()
            d[i] = obj_name
    return d


def run(sequence_dir, result_file, show_false_alarms=False, detection_file=None,
        update_ms=None, video_filename=None, top_k_tracks=None, top_k_tracks_criterion='length'):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    show_false_alarms : Optional[bool]
        If True, false alarms are highlighted as red boxes.
    detection_file : Optional[str]
        Path to the detection file.
    update_ms : Optional[int]
        Number of milliseconds between cosecutive frames. Defaults to (a) the
        frame rate specifid in the seqinfo.ini file or DEFAULT_UDPATE_MS ms if
        seqinfo.ini is not available.
    video_filename : Optional[Str]
        If not None, a video of the tracking results is written to this file.
    top_k_tracks: Optional[int]
        If not None, filter final output results for longest k tracks.
    top_k_tracks_criterion: str
        If top_k_tracks is not None, then use this criterion to select top k tracks. Options are: length and
        accumulated_score.
    """
    # seq_info = deep_sort_app.gather_sequence_info(sequence_dir, detection_file)
    if 'detr' in result_file:
        track_cls_to_track_name = {i: cls_name for i, cls_name in zip(range(len(COCO_CLASSES)), COCO_CLASSES)}
        feature_dim = 256
    else:
        track_names_filepath = './resources/objects_vocab.txt'
        track_cls_to_track_name = create_id_to_name_dict(track_names_filepath)
    seq_info = deep_sort_app.gather_mpii_cooking_2_info(sequence_dir, detection_file, feature_dim=feature_dim)
    results = np.loadtxt(result_file, delimiter=',')
    if top_k_tracks is not None:
        results = postprocessing.filter_top_k_longest_tracks(results, top_k_tracks, top_k_tracks_criterion)

    if show_false_alarms and seq_info["groundtruth"] is None:
        raise ValueError("No groundtruth available. Cannot show false alarms.")

    def frame_callback(vis, frame_idx):
        print("Frame idx", frame_idx)
        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        vis.set_image(image.copy())

        if seq_info["detections"] is not None:
            detections = deep_sort_app.create_detections(
                seq_info["detections"], frame_idx)
            vis.draw_detections(detections)

        mask = results[:, 0].astype(np.int) == frame_idx
        track_ids = results[mask, 1].astype(np.int)
        track_classes = results[mask, 2].astype(np.int)
        track_classes = [track_cls_to_track_name[track_class] for track_class in track_classes]
        boxes = results[mask, 4:8]
        vis.draw_groundtruth(track_ids, boxes, track_classes)

        if show_false_alarms:
            groundtruth = seq_info["groundtruth"]
            mask = groundtruth[:, 0].astype(np.int) == frame_idx
            gt_boxes = groundtruth[mask, 2:6]
            for box in boxes:
                # NOTE(nwojke): This is not strictly correct, because we don't
                # solve the assignment problem here.
                min_iou_overlap = 0.5
                if iou(box, gt_boxes).max() < min_iou_overlap:
                    vis.viewer.color = 0, 0, 255
                    vis.viewer.thickness = 4
                    vis.viewer.rectangle(*box.astype(np.int))

    if update_ms is None:
        update_ms = seq_info["update_ms"]
    if update_ms is None:
        update_ms = DEFAULT_UPDATE_MS
    visualizer = visualization.Visualization(seq_info, update_ms)
    if video_filename is not None:
        visualizer.viewer.enable_videowriter(video_filename)
    visualizer.run(frame_callback)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--sequence_dir", help="Path to the MOTChallenge sequence directory.",
        default=None, required=True)
    parser.add_argument(
        "--result_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections (optional).",
        default=None)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    parser.add_argument(
        "--output_file", help="Filename of the (optional) output video.",
        default=None)
    parser.add_argument(
        "--show_false_alarms", help="Show false alarms as red bounding boxes.",
        type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.result_file, args.show_false_alarms,
        args.detection_file, args.update_ms, args.output_file)
