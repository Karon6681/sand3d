"""Extrude up to a bound, and the hand-built parts: closed, the right size."""
import math
import unittest

import sand3d as s3d
from test_solids import closed_volume


class UpTo(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_up_to_a_plane(self):
        a = s3d.extrude("a", s3d.rect(10, 10), until=s3d.Frame((0, 0, 20), (0, 0.5, 1)))
        self.assertEqual(closed_volume(a)[0], True)
        self.assertAlmostEqual(closed_volume(a)[1], 2000, places=6)
        b = s3d.extrude("b", s3d.rect(10, 10), until=s3d.Frame((0, 0, -15)))
        self.assertAlmostEqual(closed_volume(b)[1], 1500, places=6)

    def test_up_to_a_curved_body(self):
        pipe = s3d.at(s3d.cylinder("pipe", 40, 100), x=-50, z=50, ry=90)
        rib = s3d.extrude("rib", s3d.rect(10, 20), until=pipe)
        closed, v = closed_volume(rib)
        lens = 2 * (5 * math.sqrt(300) + 200 * math.asin(0.5))
        self.assertTrue(closed)
        self.assertAlmostEqual(v, 10 * (1000 - lens), delta=0.005 * v)

    def test_up_to_needs_a_target_ahead(self):
        far = s3d.at(s3d.box("far", 10, 10, 10), x=500)
        self.assertRaises(ValueError, s3d.extrude, "x", s3d.rect(5, 5), until=far)


class Parts(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_gear(self):
        g = s3d.gear("g", 20, 2, 10, bore=8)
        closed, v = closed_volume(g)
        self.assertTrue(closed)
        radii = [math.hypot(p.co.x, p.co.y) for p in g.data.vertices]
        self.assertAlmostEqual(max(radii), 22, places=4)          # tip: pitch radius + module
        self.assertAlmostEqual(min(r for r in radii if r > 4.5), 17.5, places=4)  # root: - 1.25 module
        self.assertTrue(math.pi * 17.5 ** 2 * 10 < v + math.pi * 16 * 10 < math.pi * 22 ** 2 * 10)

    def test_thread_and_knurl(self):
        t = s3d.thread("t", 10, 20, 1.5)
        closed, v = closed_volume(t)
        self.assertTrue(closed)
        minor = 10 - 2 * 0.6134 * 1.5
        self.assertTrue(math.pi * (minor / 2) ** 2 * 20 < v < math.pi * 25 * 20)
        for k in (s3d.knurl("k", 20, 30, 1), s3d.knurl("s", 20, 30, 1, straight=True)):
            closed, v = closed_volume(k)
            self.assertTrue(closed)
            self.assertTrue(math.pi * 9.5 ** 2 * 30 < v < math.pi * 100 * 30)


if __name__ == "__main__":
    unittest.main()
