import pandas as pd

from medwatch.core.events import Severity
from medwatch.video.objects import Detection, count_findings, roi_findings
from medwatch.video.rules import Roi

ROI = Roi(
    name="area_restrita",
    polygon=[(0.0, 0.0), (0.2, 0.0), (0.2, 1.0), (0.0, 1.0)],
    forbidden_classes=["person"],
    message="Pessoa em área restrita",
)


def det(label, x1, y1, x2, y2):
    return Detection(label=label, confidence=0.9, bbox=(x1, y1, x2, y2))


def test_person_anchor_is_the_feet():
    person = det("person", 0, 0, 100, 200)
    assert person.anchor(400, 400) == (0.125, 0.5)
    assert det("bottle", 0, 0, 100, 200).anchor(400, 400) == (0.125, 0.25)


def test_roi_invasion_with_minimum_duration():
    inside = [det("person", 10, 100, 50, 300)]
    outside = [det("person", 300, 100, 350, 300)]
    other = [det("chair", 10, 100, 50, 300)]  # classe permitida
    frames = [outside, inside, inside, inside, inside, outside, other, other, other]
    times = pd.Series([i / 10 for i in range(len(frames))])
    [finding] = roi_findings(ROI, frames, times, (400, 400), min_seconds=0.3)
    assert finding.type == "roi_area_restrita"
    assert (finding.start, finding.end) == (0.1, 0.4)
    assert finding.severity is Severity.HIGH


def test_instrument_count_mismatch():
    times = pd.Series([i / 10 for i in range(60)])
    two = [det("scissors", 0, 0, 10, 10), det("scissors", 20, 0, 30, 10)]
    one = [det("scissors", 0, 0, 10, 10)]
    frames = [two] * 30 + [one] * 30
    [finding] = count_findings(["scissors"], frames, times, edge_seconds=1.0)
    assert finding.evidence == {"class": "scissors", "count_start": 2, "count_end": 1}
    assert count_findings(["scissors"], [two] * 60, times) == []
