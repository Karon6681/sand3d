"""`station`: a loft section at height z, exactly `at(superellipse(w, d, n=n), x=x, y=y, z=z)`."""
import unittest

import bpy  # noqa: F401

import sand3d as s3d
from harness import digest


class Station(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_is_a_placed_superellipse(self):
        for args, kw in [((0, 40, 30), {}), ((25, 48, 55), {"x": 11, "y": -35, "n": 4}), ((-5, 10, 4), {"n": 2.5})]:
            z, w, d = args
            got = s3d.station(*args, **kw)
            want = s3d.at(s3d.superellipse(w, d, n=kw.get("n", 2)), x=kw.get("x", 0), y=kw.get("y", 0), z=z)
            self.assertEqual(tuple(got.matrix), tuple(want.matrix))
            self.assertEqual(got.profile.loops[0].pts, want.profile.loops[0].pts)

    def test_lofts_as_the_placed_superellipses_do(self):
        rows = [(0, 34, 38, 10, 34, 2), (16, 48, 55, 11, 35, 2), (60, 40, 44, 14, 38, 4)]
        s3d.loft("a", *[s3d.station(z=z, w=a, d=b, x=x, y=y, n=n) for z, a, b, x, y, n in rows])
        by_station = digest()
        s3d.empty_scene()
        s3d.loft("a", *[s3d.at(s3d.superellipse(a, b, n=n), x=x, y=y, z=z) for z, a, b, x, y, n in rows])
        self.assertEqual(by_station, digest())

    def test_taper_as_superellipse(self):
        """An egg section; taper=0 is 1.2.0's station, to the bit."""
        rows = [(0, 300, 900, 0, 2350, 2.5, 0.2), (400, 700, 1500, 0, 2050, 2.5, 0.35), (900, 500, 1200, 0, 2100, 2, 0)]
        for z, w, d, x, y, n, t in rows:
            got = s3d.station(z, w, d, x=x, y=y, n=n, taper=t)
            want = s3d.at(s3d.superellipse(w, d, n=n, taper=t), x=x, y=y, z=z)
            self.assertEqual(tuple(got.matrix), tuple(want.matrix))
            self.assertEqual(got.profile.loops[0].pts, want.profile.loops[0].pts)
        plain = s3d.station(25, 48, 55, x=11, y=-35, n=4)
        zero = s3d.station(25, 48, 55, x=11, y=-35, n=4, taper=0)
        old = s3d.at(s3d.superellipse(48, 55, n=4), x=11, y=-35, z=25)       # 1.2.0's own definition
        for p in (plain, zero):
            self.assertEqual(tuple(p.matrix), tuple(old.matrix))
            self.assertEqual(p.profile.loops[0].pts, old.profile.loops[0].pts)
        egg = s3d.loft("cabin", *[s3d.station(z, w, d, x=x, y=y, n=n, taper=t) for z, w, d, x, y, n, t in rows])
        ys = [v.co.y for v in egg.data.vertices if abs(v.co.z - 400) < 1e-6]
        self.assertAlmostEqual(max(ys) - min(ys), 1500, delta=2)   # 96 points by arc length: not on the tip

    def test_one_nothing_used_is_said_as_for_at(self):
        s3d.station(10, 40, 30)
        with self.assertRaisesRegex(ValueError, "nothing used"):
            s3d.box("b", 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
