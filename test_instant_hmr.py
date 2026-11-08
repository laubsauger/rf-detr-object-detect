"""Focused adapter checks; no model weights or GPU required."""
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np

from instant_hmr import MeshModel, mesh_options
from setup_instanthmr import SOURCE, resolve_assets, validate_fit

sys.path.insert(0, str(SOURCE))
from instanthmr.smoothing import PoseFilter


class AdapterTests(unittest.TestCase):
    def raw_model(self):
        model = MeshModel.__new__(MeshModel)
        model.options = mesh_options({"smoothing": "off"})
        model.smoother = None
        return model

    def test_person_selection_colour_order_and_limit(self):
        rgb = np.full((2, 4, 3), [10, 20, 30], dtype=np.uint8)
        classes = np.array([0, 1, 0, 0, 0, 0])
        confidence = np.array([.6, .99, .7, .95, .8, .9])
        boxes = np.arange(24, dtype=np.float32).reshape(6, 4)
        model = self.raw_model()

        def detect(bgr, threshold):
            np.testing.assert_array_equal(bgr[0, 0], [30, 20, 10])
            self.assertEqual(threshold, .5)
            return SimpleNamespace(class_id=classes, confidence=confidence, xyxy=boxes)

        def predict(image, selected, scores, timestamp):
            self.assertGreater(timestamp, 0)
            self.assertTrue(image.flags.c_contiguous)
            np.testing.assert_array_equal(image, rgb)
            np.testing.assert_array_equal(selected, boxes[[3, 5, 4, 2]])
            np.testing.assert_array_equal(scores, confidence[[3, 5, 4, 2]])
            return ["person"]

        model.detector = SimpleNamespace(predict=detect)
        model.hmr = SimpleNamespace(predict=predict, pose_filter=None)
        persons, timings = model.predict_mesh(rgb, .5)
        self.assertEqual(persons, ["person"])
        self.assertEqual(set(timings), {"detector", "hmr"})

    def test_no_people_passes_empty_boxes(self):
        model = self.raw_model()
        model.detector = SimpleNamespace(predict=lambda *args: SimpleNamespace(
            class_id=np.array([1]), confidence=np.array([.9]), xyxy=np.zeros((1, 4))))

        def predict(image, boxes, scores, timestamp):
            self.assertEqual(boxes.shape, (0, 4))
            self.assertEqual(scores.shape, (0,))
            return []

        model.hmr = SimpleNamespace(predict=predict, pose_filter=None)
        self.assertEqual(model.predict_mesh(np.zeros((2, 2, 3), np.uint8), .5)[0], [])

    def test_filters_are_private_to_each_session_and_reset(self):
        cached = MeshModel.__new__(MeshModel)
        network, body = object(), object()
        cached.hmr = SimpleNamespace(network=network, body=body, pose_filter=None)
        first, second = cached.for_session({}), cached.for_session({})
        self.assertIs(first.hmr.network, second.hmr.network)
        self.assertIs(first.hmr.body, second.hmr.body)
        self.assertIsNone(cached.hmr.pose_filter)
        self.assertIsNot(first.hmr.pose_filter, second.hmr.pose_filter)
        boxes = np.array([[0, 0, 100, 100]])
        zero, moved = np.zeros((1, 204), np.float32), np.ones((1, 204), np.float32) * .2
        first.hmr.pose_filter(boxes, zero, 10.0)
        filtered = first.hmr.pose_filter(boxes, moved, 10.01)
        self.assertLess(filtered[0, 10], moved[0, 10])
        np.testing.assert_array_equal(second.hmr.pose_filter(boxes, moved, 10.01), moved)
        first.reset_tracking()
        np.testing.assert_array_equal(first.hmr.pose_filter(boxes, moved, 10.02), moved)
        first.configure({"smoothing": "output", "betaScale": .5, "maxPersons": 1})
        self.assertIsNone(first.hmr.pose_filter)
        self.assertEqual(first.smoother.beta_scale, .5)
        self.assertEqual(first.options["maxPersons"], 1)
        first.configure({"smoothing": "off"})
        self.assertIsNone(first.hmr.pose_filter)
        self.assertIsNone(first.smoother)

    def test_missing_people_expire_pose_history(self):
        model = self.raw_model()
        pose_filter = PoseFilter()
        model.hmr = SimpleNamespace(predict=lambda *args, **kwargs: [], pose_filter=pose_filter)
        model.detector = SimpleNamespace(predict=lambda *args: SimpleNamespace(
            class_id=np.array([], int), confidence=np.array([]), xyxy=np.empty((0, 4))))
        boxes = np.array([[0, 0, 100, 100]])
        pose_filter(boxes, np.zeros((1, 204), np.float32), 1.0)
        for _ in range(6):
            model.predict_mesh(np.zeros((2, 2, 3), np.uint8), .5)
        moved = np.ones((1, 204), np.float32)
        np.testing.assert_array_equal(pose_filter(boxes, moved, 100.0), moved)

    def test_filters_follow_boxes_when_confidence_order_changes(self):
        boxes = np.array([[0, 0, 100, 100], [200, 0, 300, 100]])
        params = np.zeros((2, 204), np.float32)
        params[1, 3:68] = 1
        for kind in ("one-euro", "alpha-beta"):
            with self.subTest(kind=kind):
                filtering = PoseFilter(kind)
                filtering(boxes, params, 1.0)
                np.testing.assert_allclose(filtering(boxes[::-1], params[::-1], 1.03), params[::-1], atol=1e-6)

    def test_invalid_options_rejected_without_changing_filter_state(self):
        cached = MeshModel.__new__(MeshModel)
        cached.hmr = SimpleNamespace(pose_filter=None)
        session = cached.for_session({})
        original = session.hmr.pose_filter
        for options in ({"smoothing": "unknown"}, {"maxPersons": 0}, {"maxPersons": True},
                        {"betaScale": float("nan")}, {"betaScale": 0}, {"betaScale": 5}, {"extra": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                session.configure(options)
            self.assertIs(session.hmr.pose_filter, original)

    def test_fit_selects_pinned_graph(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=lambda repo, filename, **kwargs: filename):
            for fit, name in [(0, "instanthmr-small.onnx"), (1, "instanthmr-small-fit1.onnx"), (2, "instanthmr-small-fit2.onnx")]:
                self.assertEqual(resolve_assets("small", fit), (name, "mhr-body.onnx"))
        for fit in (-1, 3, True, "1"):
            with self.assertRaises(ValueError):
                validate_fit(fit)


if __name__ == "__main__":
    unittest.main()
