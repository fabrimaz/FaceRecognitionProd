import dlib
import numpy as np
import cv2
import pickle
import json


def find_best_bounding_box(candidate_bounding_boxes, gray_frame):
    # computes the size of the bounding box diagonal
    mean_sizes = (
            np.sum(
                np.array(
                    [
                        [rect.top() - rect.bottom(), rect.left() - rect.right()]
                        for rect in candidate_bounding_boxes
                    ]
                )
                ** 2,
                axis=-1,
            )
            ** 0.5
    )

    # computes the position of the middle of bounding boxes with respect to the middle of the image
    mean_points = np.array(
        [
            [(rect.top() + rect.bottom()) / 2.0, (rect.left() + rect.right()) / 2.0]
            for rect in candidate_bounding_boxes
        ]
    ) - np.array([gray_frame.shape[0] / 2.0, gray_frame.shape[1] / 2.0])

    # computes the distances to center, divided by the bounding box diagonal
    prop_dist = np.sum(mean_points ** 2, axis=-1) ** 0.5 / mean_sizes

    # gets the closer bounding box to the center
    best_bounding_box_id = np.argmin(prop_dist)

    # compute best bounding box
    best_bounding_box = dlib.rectangle(
        int(candidate_bounding_boxes[best_bounding_box_id].left()),
        int(candidate_bounding_boxes[best_bounding_box_id].top()),
        int(candidate_bounding_boxes[best_bounding_box_id].right()),
        int(candidate_bounding_boxes[best_bounding_box_id].bottom()),
    )

    return best_bounding_box


def compress(frame: np.array, compression_factor: float) -> np.array:
    """
    Compress the reference image can affect the efficiency of the matching
    :param compression_factor: if < 1 increase size else reduce size of image
    """
    compressed_shape = (
        int(frame.shape[1] / compression_factor),
        int(frame.shape[0] / compression_factor),
    )
    frame = cv2.resize(frame, compressed_shape)

    return frame


def encodingsRead(path):
    data = pickle.loads(open(path, "rb").read())
    return data


def readTrueName(rec_name):
    with open('./src/labelled_videos.json') as json_file:
        data_json = json.load(json_file)

    if data_json is None:
        return "Error"
    for real_name in data_json:
        if rec_name in data_json[real_name]:
            return real_name
    return "unknown"
