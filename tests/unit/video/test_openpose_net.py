import numpy as np
import torch

from medwatch.video.pose.openpose_net import (
    BodyPoseModel,
    find_peaks,
    infer_maps,
    people_keypoints,
)


def test_model_has_original_layer_names_and_output_channels():
    model = BodyPoseModel().eval()
    names = set(model.state_dict())
    # os pesos da CMU são carregados pelo nome da camada: a arquitetura precisa bater
    assert "model0.conv1_1.weight" in names
    assert "model6_2.Mconv7_stage6_L2.weight" in names
    assert len(names) == 184
    with torch.no_grad():
        paf, heat = model(torch.zeros(1, 3, 64, 64))
    assert paf.shape == (1, 38, 8, 8)
    assert heat.shape == (1, 19, 8, 8)
    # a última camada de heatmaps não tem ReLU (como no modelo Caffe original)
    assert "relu_Mconv7_stage6_L2" not in dict(model.model6_2.named_children())


def test_infer_maps_returns_image_sized_maps():
    model = BodyPoseModel().eval()
    heat, paf = infer_maps(model, np.zeros((120, 90, 3), dtype=np.uint8), box_size=64)
    assert heat.shape == (120, 90, 19)
    assert paf.shape == (120, 90, 38)


def test_find_peaks_locates_heatmap_maxima():
    heat = np.zeros((100, 100, 19), dtype=np.float32)
    yy, xx = np.mgrid[:100, :100]
    heat[:, :, 0] = np.exp(-((xx - 30) ** 2 + (yy - 60) ** 2) / 50)
    peaks = find_peaks(heat, threshold=0.1)
    assert [(p[0], p[1]) for p in peaks[0]] == [(30.0, 60.0)]
    assert all(not p for p in peaks[1:18])


def test_people_keypoints_reads_subset():
    candidate = np.array([[10, 20, 0.9, 0], [30, 40, 0.8, 1]])
    subset = -np.ones((1, 20))
    subset[0, 0], subset[0, 1] = 0, 1
    subset[0, 18], subset[0, 19] = 3.4, 2
    [(points, score)] = people_keypoints(candidate, subset)
    assert points[0].tolist() == [10, 20, 0.9]
    assert points[1].tolist() == [30, 40, 0.8]
    assert points[2].tolist() == [0, 0, 0]
    assert score == 1.7


def _draw_limb(pafs, a, b, channels, width=2):
    """Preenche o PAF de um membro: vetor unitário de a para b ao longo do segmento."""
    vec = np.subtract(b, a) / np.linalg.norm(np.subtract(b, a))
    for t in np.linspace(0, 1, 50):
        x, y = (np.array(a) + t * np.subtract(b, a)).round().astype(int)
        pafs[y - width : y + width + 1, x - width : x + width + 1, channels[0]] = vec[0]
        pafs[y - width : y + width + 1, x - width : x + width + 1, channels[1]] = vec[1]


def test_group_people_links_parts_of_two_people_separately():
    from medwatch.video.pose.openpose_net import PAF_IDX, group_people

    pafs = np.zeros((200, 300, 38), dtype=np.float32)
    # pescoço(1) → ombro dir.(2) → cotovelo(3) → punho(4), para duas pessoas lado a lado
    people = [
        [(50, 50), (50, 80), (50, 110), (50, 140)],
        [(200, 50), (200, 80), (200, 110), (200, 140)],
    ]
    limbs = [(0, PAF_IDX[0]), (1, PAF_IDX[2]), (2, PAF_IDX[3])]  # (2,3), (3,4), (4,5)
    for joints in people:
        for start, (cx, cy) in limbs:
            _draw_limb(pafs, joints[start], joints[start + 1], (cx - 19, cy - 19))
    # como no find_peaks: o id de cada pico é sua posição na lista achatada por parte
    peaks: list[list[tuple[float, ...]]] = [[] for _ in range(18)]
    counter = 0
    for part in range(1, 5):
        for joints in people:
            x, y = joints[part - 1]
            peaks[part].append((float(x), float(y), 0.9, float(counter)))
            counter += 1

    candidate, subset = group_people(peaks, pafs, image_height=200)
    assert len(subset) == 2
    for person in subset:
        assert person[19] == 4  # quatro partes ligadas
        xs = {candidate[int(i), 0] for i in person[1:5]}
        assert len(xs) == 1  # todas as partes da mesma pessoa (mesmo x)


def test_group_people_discards_isolated_parts():
    from medwatch.video.pose.openpose_net import group_people

    peaks: list[list[tuple[float, ...]]] = [[] for _ in range(18)]
    peaks[0].append((10.0, 10.0, 0.9, 0.0))
    candidate, subset = group_people(peaks, np.zeros((50, 50, 38), dtype=np.float32), 50)
    assert len(candidate) == 1
    assert len(subset) == 0
