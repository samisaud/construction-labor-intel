from __future__ import annotations

import pytest

from app.inference_engine import _associate_ppe, _bbox_iou
from app.schemas import PPEStatus

                                                     
PERSON = (100.0, 100.0, 200.0, 400.0)

def _hardhat_at_head() -> tuple[float, float, float, float]:
                                              
    return (110.0, 110.0, 190.0, 180.0)

def _vest_at_torso() -> tuple[float, float, float, float]:
                                                  
    return (105.0, 200.0, 195.0, 320.0)

def _hardhat_at_feet() -> tuple[float, float, float, float]:
                                               
    return (110.0, 350.0, 190.0, 400.0)

class TestBboxIoU:
    def test_identical_boxes(self):
        b = (0.0, 0.0, 10.0, 10.0)
        assert _bbox_iou(b, b) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _bbox_iou((0, 0, 5, 5), (100, 100, 105, 105)) == 0.0

    def test_half_overlap(self):
                                                         
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
                                                                    
        assert _bbox_iou(a, b) == pytest.approx(1 / 3)

class TestPPEAssociation:
    def test_compliant_full_ppe(self):
        result = _associate_ppe(PERSON, [_hardhat_at_head()], [_vest_at_torso()])
        assert result == PPEStatus.COMPLIANT

    def test_no_ppe_at_all(self):
        result = _associate_ppe(PERSON, [], [])
        assert result == PPEStatus.BOTH_VIOLATION

    def test_hardhat_only_means_torso_violation(self):
        result = _associate_ppe(PERSON, [_hardhat_at_head()], [])
        assert result == PPEStatus.TORSO_VIOLATION

    def test_vest_only_means_head_violation(self):
        result = _associate_ppe(PERSON, [], [_vest_at_torso()])
        assert result == PPEStatus.HEAD_VIOLATION

    def test_hardhat_at_feet_does_not_count(self):
                                                                           
        result = _associate_ppe(PERSON, [_hardhat_at_feet()], [_vest_at_torso()])
        assert result == PPEStatus.HEAD_VIOLATION

    def test_zero_height_person_returns_unknown(self):
        bad = (100.0, 100.0, 200.0, 100.0)              
        assert _associate_ppe(bad, [_hardhat_at_head()], [_vest_at_torso()]) == PPEStatus.UNKNOWN

    def test_multiple_hardhats_any_match_counts(self):
                                                                                 
        result = _associate_ppe(
            PERSON,
            [_hardhat_at_feet(), _hardhat_at_head()],
            [_vest_at_torso()],
        )
        assert result == PPEStatus.COMPLIANT
