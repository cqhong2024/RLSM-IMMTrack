import copy
import types
import unittest
from unittest.mock import patch

import numpy as np

from rlsm_immtrack.imm import BoxIMM, box_from_state, state_from_box
from rlsm_immtrack.safecl import AdaptiveBoxIMM, ClosedLoopTracker
from rlsm_immtrack.tracker import RiskLimitedDualMemory, RiskLimitedShadowTracker


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.box = np.array([20., 20., 24., 24.])
        self.image = np.random.RandomState(7).randint(0, 255, (80, 80, 3)).astype(np.uint8)

    def test_box_conversion(self):
        np.testing.assert_allclose(box_from_state(state_from_box(self.box, 80, 80), 80, 80),
                                   self.box)

    def test_imm_stability(self):
        imm = AdaptiveBoxIMM(self.box, 80, 80, False)
        for i in range(100):
            imm.predict()
            imm.correct(self.box + [i * .1, 0, 0, 0], .8)
            self.assertAlmostEqual(imm.probabilities.sum(), 1.)
            self.assertTrue((imm.probabilities >= 0).all())
            for covariance in imm.covariances:
                self.assertTrue(np.isfinite(covariance).all())
                self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)

    def test_low_confidence_does_not_correct(self):
        imm = AdaptiveBoxIMM(self.box, 80, 80, False)
        imm.predict()
        previous = [state.copy() for state in imm.states]
        _, _, updated = imm.correct(self.box + 100, .01)
        self.assertFalse(updated)
        for a, b in zip(previous, imm.states):
            np.testing.assert_array_equal(a, b)

    def test_memory_requires_future_support(self):
        memory = RiskLimitedDualMemory(self.image, self.box, memory_supports=3)
        memory.observe(self.image, self.box, .9, 5, True)
        self.assertEqual(memory.provisional_commits, 0)
        self.assertEqual(memory.pending["supports"], 0)
        for frame in (6, 7):
            memory.observe(self.image, self.box, .9, frame, True)
            self.assertEqual(memory.provisional_commits, 0)
        memory.observe(self.image, self.box, .9, 8, True)
        self.assertEqual(memory.provisional_commits, 1)
        self.assertEqual(len(memory.short), 2)
        self.assertTrue(memory.long[0]["initial"])

    def test_quarantine_clears_proposal(self):
        memory = RiskLimitedDualMemory(self.image, self.box)
        memory.observe(self.image, self.box, .9, 5, True)
        memory.quarantine()
        self.assertIsNone(memory.pending)
        self.assertEqual(memory.rejections, 1)

    def test_shadow_output_does_not_write_main_state(self):
        base = types.SimpleNamespace(
            template_list=[object()], template_anno_list=[object()], state=self.box.copy())
        tracker = RiskLimitedShadowTracker(
            base, self.image, self.box, True, True, False, 6, 4, entry_streak=1)
        imm_before = copy.deepcopy(tracker.imm.states)
        baseline = self.box.copy()
        shadow = self.box + [8., 0., 0., 0.]
        post_templates, post_annotations = [object()], [object()]
        tracker.visual_probe = lambda *args: {
            "box": shadow, "network_score": .95, "quality": .95, "psr": 30.,
        }
        tracker.shadow_memory.score = lambda *args: (.8, {
            "long_score": .8, "short_score": .8, "agreement": 1.})
        observed = []
        tracker.shadow_memory.observe = lambda image, box, *args: (
            observed.append(box.copy()) or
            {"long_score": .8, "short_score": .8, "agreement": 1.})

        def safe_step(instance, image):
            instance.frame_id += 1
            base.state = baseline.copy()
            base.template_list = post_templates
            base.template_anno_list = post_annotations
            return baseline, {
                "center_sigma_px": 5., "quality": .2, "network_score": .2,
                "psr": 1., "measurement_updated": True, "visual_seconds": 0., "probes": 1,
            }

        with patch.object(ClosedLoopTracker, "track", safe_step), \
                patch("torch.cuda.synchronize", lambda: None):
            output, diagnostic = tracker.track(self.image)
        self.assertTrue(diagnostic["selected_shadow"])
        np.testing.assert_array_equal(output, shadow)
        np.testing.assert_array_equal(base.state, baseline)
        np.testing.assert_array_equal(observed[0], baseline)
        self.assertIs(base.template_list, post_templates)
        self.assertIs(base.template_anno_list, post_annotations)
        for a, b in zip(imm_before, tracker.imm.states):
            np.testing.assert_array_equal(a, b)

    def test_premeasurement_shadow_prediction_holds_size(self):
        base = types.SimpleNamespace()
        tracker = RiskLimitedShadowTracker(base, self.image, self.box, True, True, False, 6, 4)
        tracker.shadow_velocity[:] = [2., -3.]
        np.testing.assert_array_equal(tracker._shadow_prediction(), self.box + [2., -3., 0., 0.])
        np.testing.assert_array_equal(tracker.shadow_anchor, self.box)


if __name__ == "__main__":
    unittest.main()
