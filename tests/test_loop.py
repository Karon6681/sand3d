"""loft(loop=): a ring; closed= reads as "capped" and is gone."""
import unittest

import sand3d as s3d
from test_solids import closed_volume


class Loop(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_loop_makes_a_ring_and_an_open_loft_is_capped(self):
        sections = lambda: [s3d.at(s3d.superellipse(10, 4), x=x, y=y, ry=-90)  # noqa: E731
                            for x, y in s3d.superellipse(100, 60).loops[0].pts[::8]]
        ring = s3d.loft("band", *sections(), follow=True, loop=True)
        closed, volume = closed_volume(ring)
        self.assertTrue(closed)
        self.assertGreater(volume, 0)
        names = {a.name for a in ring.data.attributes}
        self.assertFalse({"s3d_face:start", "s3d_face:end"} & names, "a ring has no ends")
        with self.assertRaises(ValueError) as caught:
            s3d.loft("r", s3d.circle(10), s3d.at(s3d.circle(10), x=50), loop=True)
        self.assertEqual(str(caught.exception), "loft: a ring (loop=True) needs three sections or more, not 2")
        capped = s3d.loft("vase", s3d.circle(40), s3d.at(s3d.circle(60), z=50))
        self.assertIn("s3d_face:start", {a.name for a in capped.data.attributes})
        self.assertTrue(closed_volume(capped)[0])

    def test_closed_says_loop(self):
        with self.assertRaises(TypeError) as caught:
            s3d.loft("band", s3d.circle(10), s3d.at(s3d.circle(10), z=10), closed=True)
        self.assertEqual(str(caught.exception), "loft: closed= is loop= in 1.0 — loop=True joins the last section "
                                                "back to the first, a ring with no ends; without it, the two ends of a "
                                                "loft are capped")

    def test_taper_and_arc_say_which_way(self):
        """The docstrings, and so api.md: a positive taper narrows, an arc runs anticlockwise."""
        self.assertIn("taper > 0 narrows it upward", s3d.extrude.__doc__.split("\n")[0])
        self.assertIn("anticlockwise if end > start", s3d.arc.__doc__.split("\n")[0])
        for taper, inward in ((10, True), (-5, False)):
            prism = s3d.extrude("p", s3d.rect(20, 20), 10, taper=taper)
            top = max(abs(v.co.x) for v in prism.data.vertices if abs(v.co.z - 10) < 1e-6)
            with self.subTest(taper=taper):
                self.assertEqual(top < 10, inward)
        self.assertGreater(s3d.arc((0, 0), 10, 0, 90).points[1][1], 0, "from +X towards +Y: anticlockwise")
        self.assertLess(s3d.arc((0, 0), 10, 0, -90).points[1][1], 0, "end < start: clockwise")


if __name__ == "__main__":
    unittest.main()
