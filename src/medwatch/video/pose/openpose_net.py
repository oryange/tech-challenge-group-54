"""Rede de corpo do OpenPose (COCO-18) em PyTorch e agrupamento de pessoas por PAFs.

Modelo: Cao et al., "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields",
CVPR 2017 (CMU). Código adaptado de pytorch-openpose (Hzzone, licença MIT) na versão
distribuída pelo controlnet_aux (licença Apache-2.0). Veja THIRD_PARTY_NOTICES.md.

Mudanças em relação à origem: código reorganizado e tipado; a última camada de heatmaps
(Mconv7_stage6_L2) ficou sem ReLU, como no modelo Caffe original (a origem repetia
Mconv7_stage6_L1 na lista); a média entre escalas foi corrigida.
"""

import math
from collections import OrderedDict

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

NUM_PARTS = 18
# pares de partes (base 1) ligados por um membro, e os canais de PAF correspondentes
LIMB_SEQ = (
    (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12),
    (12, 13), (13, 14), (2, 1), (1, 15), (15, 17), (1, 16), (16, 18), (3, 17), (6, 18),
)  # fmt: skip
PAF_IDX = (
    (31, 32), (39, 40), (33, 34), (35, 36), (41, 42), (43, 44), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (47, 48), (49, 50), (53, 54), (51, 52),
    (55, 56), (37, 38), (45, 46),
)  # fmt: skip
NO_RELU = {f"Mconv7_stage{i}_L{b}" for i in range(2, 7) for b in (1, 2)} | {
    "conv5_5_CPM_L1",
    "conv5_5_CPM_L2",
}


def _make_layers(block: "OrderedDict[str, list[int]]") -> nn.Sequential:
    layers: list[tuple[str, nn.Module]] = []
    for name, v in block.items():
        if "pool" in name:
            layers.append((name, nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])))
            continue
        layers.append((name, nn.Conv2d(v[0], v[1], kernel_size=v[2], stride=v[3], padding=v[4])))
        if name not in NO_RELU:
            layers.append((f"relu_{name}", nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


def _stage_block(stage: int, branch: int, out_channels: int) -> "OrderedDict[str, list[int]]":
    s, b = stage, branch
    return OrderedDict(
        [
            (f"Mconv1_stage{s}_L{b}", [185, 128, 7, 1, 3]),
            (f"Mconv2_stage{s}_L{b}", [128, 128, 7, 1, 3]),
            (f"Mconv3_stage{s}_L{b}", [128, 128, 7, 1, 3]),
            (f"Mconv4_stage{s}_L{b}", [128, 128, 7, 1, 3]),
            (f"Mconv5_stage{s}_L{b}", [128, 128, 7, 1, 3]),
            (f"Mconv6_stage{s}_L{b}", [128, 128, 1, 1, 0]),
            (f"Mconv7_stage{s}_L{b}", [128, out_channels, 1, 1, 0]),
        ]
    )


class BodyPoseModel(nn.Module):
    """VGG-19 inicial + 6 estágios com dois ramos: PAFs (L1, 38 canais) e heatmaps (L2, 19)."""

    def __init__(self) -> None:
        super().__init__()
        self.model0 = _make_layers(
            OrderedDict(
                [
                    ("conv1_1", [3, 64, 3, 1, 1]),
                    ("conv1_2", [64, 64, 3, 1, 1]),
                    ("pool1_stage1", [2, 2, 0]),
                    ("conv2_1", [64, 128, 3, 1, 1]),
                    ("conv2_2", [128, 128, 3, 1, 1]),
                    ("pool2_stage1", [2, 2, 0]),
                    ("conv3_1", [128, 256, 3, 1, 1]),
                    ("conv3_2", [256, 256, 3, 1, 1]),
                    ("conv3_3", [256, 256, 3, 1, 1]),
                    ("conv3_4", [256, 256, 3, 1, 1]),
                    ("pool3_stage1", [2, 2, 0]),
                    ("conv4_1", [256, 512, 3, 1, 1]),
                    ("conv4_2", [512, 512, 3, 1, 1]),
                    ("conv4_3_CPM", [512, 256, 3, 1, 1]),
                    ("conv4_4_CPM", [256, 128, 3, 1, 1]),
                ]
            )
        )
        for branch, out_channels in ((1, 38), (2, 19)):
            stage1 = OrderedDict(
                [
                    (f"conv5_1_CPM_L{branch}", [128, 128, 3, 1, 1]),
                    (f"conv5_2_CPM_L{branch}", [128, 128, 3, 1, 1]),
                    (f"conv5_3_CPM_L{branch}", [128, 128, 3, 1, 1]),
                    (f"conv5_4_CPM_L{branch}", [128, 512, 1, 1, 0]),
                    (f"conv5_5_CPM_L{branch}", [512, out_channels, 1, 1, 0]),
                ]
            )
            setattr(self, f"model1_{branch}", _make_layers(stage1))
            for stage in range(2, 7):
                block = _stage_block(stage, branch, out_channels)
                setattr(self, f"model{stage}_{branch}", _make_layers(block))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model0(x)
        out = features
        for stage in range(1, 7):
            paf = getattr(self, f"model{stage}_1")(out)
            heat = getattr(self, f"model{stage}_2")(out)
            out = torch.cat([paf, heat, features], 1)
        return paf, heat


def load_body_model(weights_path: str, device: torch.device) -> BodyPoseModel:
    model = BodyPoseModel()
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    # os pesos usam os nomes das camadas sem o prefixo do bloco (ex.: "conv1_1.weight")
    state = {name: weights[name.split(".", 1)[1]] for name in model.state_dict()}
    model.load_state_dict(state)
    return model.to(device).eval()


def _pad_right_down(img: np.ndarray, stride: int, pad_value: int) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_down = (stride - h % stride) % stride
    pad_right = (stride - w % stride) % stride
    padded = cv2.copyMakeBorder(
        img, 0, pad_down, 0, pad_right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3
    )
    return padded, pad_down, pad_right


def infer_maps(
    model: BodyPoseModel,
    image_bgr: np.ndarray,
    scales: tuple[float, ...] = (0.5,),
    box_size: int = 368,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Heatmaps (H, W, 19) e PAFs (H, W, 38) no tamanho da imagem original."""
    device = next(model.parameters()).device
    h, w = image_bgr.shape[:2]
    heatmap_avg = np.zeros((h, w, 19), dtype=np.float32)
    paf_avg = np.zeros((h, w, 38), dtype=np.float32)
    for scale in scales:
        factor = scale * box_size / h
        resized = cv2.resize(image_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        padded, pad_down, pad_right = _pad_right_down(resized, stride, 128)
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 256 - 0.5
        ).to(device)
        with torch.no_grad():
            paf, heat = model(tensor)
        for output, accumulator in ((heat, heatmap_avg), (paf, paf_avg)):
            maps = output[0].cpu().numpy().transpose(1, 2, 0)
            maps = cv2.resize(maps, None, fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            maps = maps[: padded.shape[0] - pad_down, : padded.shape[1] - pad_right]
            accumulator += cv2.resize(maps, (w, h), interpolation=cv2.INTER_CUBIC) / len(scales)
    return heatmap_avg, paf_avg


def find_peaks(heatmaps: np.ndarray, threshold: float = 0.1) -> list[list[tuple[float, ...]]]:
    """Máximos locais de cada heatmap: lista por parte de (x, y, score, id)."""
    all_peaks: list[list[tuple[float, ...]]] = []
    counter = 0
    for part in range(NUM_PARTS):
        original = heatmaps[:, :, part]
        smooth = gaussian_filter(original, sigma=3)
        is_peak = smooth > threshold
        is_peak[1:, :] &= smooth[1:, :] >= smooth[:-1, :]
        is_peak[:-1, :] &= smooth[:-1, :] >= smooth[1:, :]
        is_peak[:, 1:] &= smooth[:, 1:] >= smooth[:, :-1]
        is_peak[:, :-1] &= smooth[:, :-1] >= smooth[:, 1:]
        ys, xs = np.nonzero(is_peak)
        peaks: list[tuple[float, ...]] = [
            (float(x), float(y), float(original[y, x]), float(counter + i))
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))
        ]
        all_peaks.append(peaks)
        counter += len(peaks)
    return all_peaks


def group_people(
    all_peaks: list[list[tuple[float, ...]]],
    pafs: np.ndarray,
    image_height: int,
    paf_threshold: float = 0.05,
    mid_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Liga as partes em pessoas usando os PAFs.

    Retorna `candidate` (N, 4: x, y, score, id) e `subset` (P, 20): índices de candidate por
    parte (−1 = ausente), score total e número de partes de cada pessoa.
    """
    connections: list[np.ndarray] = []
    missing_limbs: set[int] = set()
    for k, ((part_a, part_b), (paf_x, paf_y)) in enumerate(zip(LIMB_SEQ, PAF_IDX, strict=True)):
        score_mid = pafs[:, :, [paf_x - 19, paf_y - 19]]
        cand_a, cand_b = all_peaks[part_a - 1], all_peaks[part_b - 1]
        if not cand_a or not cand_b:
            missing_limbs.add(k)
            connections.append(np.zeros((0, 5)))
            continue
        candidates = []
        for i, a in enumerate(cand_a):
            for j, b in enumerate(cand_b):
                vec = np.subtract(b[:2], a[:2])
                norm = max(0.001, math.hypot(vec[0], vec[1]))
                vec = vec / norm
                xs = np.linspace(a[0], b[0], num=mid_samples).round().astype(int)
                ys = np.linspace(a[1], b[1], num=mid_samples).round().astype(int)
                sampled = score_mid[ys, xs]
                alignment = sampled[:, 0] * vec[0] + sampled[:, 1] * vec[1]
                score = alignment.mean() + min(0.5 * image_height / norm - 1, 0)
                if (alignment > paf_threshold).sum() > 0.8 * mid_samples and score > 0:
                    candidates.append((i, j, score))
        candidates.sort(key=lambda c: c[2], reverse=True)
        connection = np.zeros((0, 5))
        for i, j, score in candidates:
            if i not in connection[:, 3] and j not in connection[:, 4]:
                link = [cand_a[i][3], cand_b[j][3], score, i, j]
                connection = np.vstack([connection, link])
                if len(connection) >= min(len(cand_a), len(cand_b)):
                    break
        connections.append(connection)

    candidate = np.array([peak for peaks in all_peaks for peak in peaks]).reshape(-1, 4)
    subset = -1 * np.ones((0, 20))
    for k, (part_a, part_b) in enumerate(LIMB_SEQ):
        if k in missing_limbs:
            continue
        index_a, index_b = part_a - 1, part_b - 1
        for parts_a, parts_b, limb_score, *_ in connections[k]:
            found = [
                j
                for j in range(len(subset))
                if subset[j][index_a] == parts_a or subset[j][index_b] == parts_b
            ][:2]
            if len(found) == 1:
                j = found[0]
                if subset[j][index_b] != parts_b:
                    subset[j][index_b] = parts_b
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[int(parts_b), 2] + limb_score
            elif len(found) == 2:
                j1, j2 = found
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                if not np.any(membership == 2):  # pessoas disjuntas: une as duas
                    subset[j1][:-2] += subset[j2][:-2] + 1
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += limb_score
                    subset = np.delete(subset, j2, 0)
                else:
                    subset[j1][index_b] = parts_b
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate[int(parts_b), 2] + limb_score
            elif k < 17:
                row = -1 * np.ones(20)
                row[index_a], row[index_b] = parts_a, parts_b
                row[-1] = 2
                row[-2] = candidate[[int(parts_a), int(parts_b)], 2].sum() + limb_score
                subset = np.vstack([subset, row])

    keep = [
        i for i in range(len(subset)) if subset[i][-1] >= 4 and subset[i][-2] / subset[i][-1] >= 0.4
    ]
    return candidate, subset[keep]


def people_keypoints(candidate: np.ndarray, subset: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Converte o resultado em arrays (18, 3) por pessoa, com o score médio por parte."""
    people = []
    for person in subset:
        points = np.zeros((NUM_PARTS, 3))
        for part, index in enumerate(person[:NUM_PARTS].astype(int)):
            if index >= 0:
                points[part] = candidate[index, :3]
        people.append((points, float(person[18] / max(person[19], 1))))
    return people
