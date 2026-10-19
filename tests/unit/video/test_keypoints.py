import json

import numpy as np

from medwatch.video.keypoints import (
    BODY25_FOR_COCO17,
    COCO17,
    KP,
    OPENPOSE18_FOR_COCO17,
    bbox_from_keypoints,
    from_openpose,
)
from medwatch.video.pose.openpose import read_openpose_json

BODY25 = [
    "nose",
    "neck",
    "r_sho",
    "r_elb",
    "r_wri",
    "l_sho",
    "l_elb",
    "l_wri",
    "mid_hip",
    "r_hip",
    "r_knee",
    "r_ank",
    "l_hip",
    "l_knee",
    "l_ank",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
    "l_bigtoe",
    "l_smalltoe",
    "l_heel",
    "r_bigtoe",
    "r_smalltoe",
    "r_heel",
]
OPENPOSE18 = [
    "nose",
    "neck",
    "r_sho",
    "r_elb",
    "r_wri",
    "l_sho",
    "l_elb",
    "l_wri",
    "r_hip",
    "r_knee",
    "r_ank",
    "l_hip",
    "l_knee",
    "l_ank",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
]
SHORT = {
    "left": "l",
    "right": "r",
    "shoulder": "sho",
    "elbow": "elb",
    "wrist": "wri",
    "hip": "hip",
    "knee": "knee",
    "ankle": "ank",
    "eye": "eye",
    "ear": "ear",
}


def short_name(coco_name: str) -> str:
    if coco_name == "nose":
        return "nose"
    side, part = coco_name.split("_")
    return f"{SHORT[side]}_{SHORT[part]}"


def test_mappings_point_to_the_same_body_part():
    for i, name in enumerate(COCO17):
        assert BODY25[BODY25_FOR_COCO17[i]] == short_name(name)
        assert OPENPOSE18[OPENPOSE18_FOR_COCO17[i]] == short_name(name)


def test_from_openpose_keeps_coordinates_and_zeroes_missing():
    points = np.zeros((25, 3))
    points[BODY25.index("l_knee")] = [100, 200, 0.8]
    points[BODY25.index("r_knee")] = [50, 60, 0.0]  # não detectado
    coco = from_openpose(points, BODY25_FOR_COCO17)
    assert coco.shape == (17, 3)
    assert coco[KP["left_knee"]].tolist() == [100, 200, 0.8]
    assert coco[KP["right_knee"]].tolist() == [0, 0, 0]


def test_bbox_ignores_low_confidence_points():
    kp = np.array([[10, 20, 0.9], [50, 80, 0.9], [500, 500, 0.05]] + [[0, 0, 0]] * 14)
    assert bbox_from_keypoints(kp) == (10, 20, 50, 80)
    assert bbox_from_keypoints(np.zeros((17, 3))) == (0, 0, 0, 0)


def test_read_openpose_json(tmp_path):
    points = np.zeros((25, 3))
    points[BODY25.index("nose")] = [120, 40, 0.9]
    points[BODY25.index("l_ank")] = [110, 300, 0.7]
    path = tmp_path / "frame_000000000000_keypoints.json"
    path.write_text(json.dumps({"people": [{"pose_keypoints_2d": points.ravel().tolist()}]}))
    people = read_openpose_json(path)
    assert len(people) == 1
    assert people[0].keypoints[KP["left_ankle"]].tolist() == [110, 300, 0.7]
    assert people[0].bbox == (110, 40, 120, 300)
