"""on= is the one word for where; a hole starts where its axis enters the part."""
import math
import unittest

import sand3d as s3d
from test_solids import closed_volume


def mouths(ob, d):
    """The heights at which the part has a ring of radius d/2 round the Z axis through (0, 0)."""
    return sorted({round(v.co.z, 4) for v in ob.data.vertices if abs(math.hypot(v.co.x, v.co.y) - d / 2) < 1e-4})


class On(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def says(self, message, call, *args, **kwargs):
        with self.assertRaises(TypeError) as caught:
            call(*args, **kwargs)
        self.assertEqual(str(caught.exception), message)

    def test_the_keywords_1_0_took_away(self):
        plate = s3d.box("plate", 40, 40, 10)
        self.says("hole: at= is on= in 1.0 — s3d.hole(ob, d, on=frames or points); on= is the one word for where",
                  s3d.hole, plate, 4, at=(0, 0))
        self.says("join: at= is gone in 1.0 — place one copy per frame with at, and join the list: "
                  "s3d.join(ob, s3d.at(tool, on=[frames]))", s3d.join, plate, s3d.box("b", 2, 2, 2),
                  at=s3d.grid(2, 1, 10))
        self.says("cut: at= is gone in 1.0 — place one copy per frame with at, and cut the list: "
                  "s3d.cut(ob, s3d.at(tool, on=[frames]))", s3d.cut, plate, s3d.box("c", 2, 2, 2),
                  at=s3d.grid(2, 1, 10))

    def test_a_hole_starts_where_its_axis_enters(self):
        """A frame of a grid without on= lies at z = 0: in the middle of a plate centred on
        it, the hole was drilled from there, into air below with depth=. It starts at the
        top now, as a point does — and from wherever the frame stands along its axis."""
        for z in (5, 25, 10, 2, 0):                           # 0: on the bottom, looking up through it
            s3d.empty_scene()
            plate = s3d.box("plate", 40, 40, 10)             # z from 0 to 10
            s3d.hole(plate, 4, on=s3d.Frame((0, 0, z)), depth=3)
            with self.subTest(frame_at=z):
                self.assertEqual(mouths(plate, 4), [7.0, 10.0])
        s3d.empty_scene()
        bracket = s3d.extrude("bracket", s3d.rect(80, 40), 6)   # from z = 0: a grid without on= lies on its bottom
        s3d.hole(bracket, 4, on=s3d.grid(1, 1), depth=3)
        self.assertEqual(mouths(bracket, 4), [3.0, 6.0])
        s3d.empty_scene()
        plate = s3d.box("plate", 40, 40, 10)
        s3d.hole(plate, 4, on=s3d.Frame((0, 0, -20), (0, 0, -1)), depth=3)   # from below, facing down
        self.assertEqual(mouths(plate, 4), [0.0, 3.0])

    def test_points_lists_and_steps(self):
        plate = s3d.box("plate", 80, 40, 10)
        s3d.cut(plate, s3d.at(s3d.box("step", 40, 50, 10), x=20, z=5))   # the right half 5 lower
        s3d.hole(plate, 4, on=[(-20, 0), (20, 0)], depth=2)
        tops = sorted({round(v.co.z, 4) for v in plate.data.vertices
                       if any(abs(math.hypot(v.co.x - x, v.co.y) - 2) < 1e-4 for x in (-20, 20))})
        self.assertEqual(tops, [3.0, 5.0, 8.0, 10.0], "each hole starts on the surface over its point")
        bracket = s3d.box("bracket", 80, 40, 6)
        s3d.hole(bracket, 5, on=s3d.grid(2, 1, 50), csk=(10, 90))   # opened from below
        rims = sorted({round(v.co.z, 4) for v in bracket.data.vertices if abs(abs(v.co.x) - 25) < 5.01
                       and abs(math.hypot(abs(v.co.x) - 25, v.co.y) - 5) < 1e-4})
        self.assertEqual(rims, [6.0], "the countersinks open on the top")
        self.assertTrue(closed_volume(bracket)[0])

    def test_inside_up_in_the_air_down_on_a_face_kept(self):
        """A U channel: a frame on the inner face of the left wall, looking at the right
        wall, stays on its face; a frame in the gap goes to the left wall's inner face;
        a frame inside the left wall goes up to that face too. Walls 5 thick: x from -15 to
        -10 and from 10 to 15, over a floor 5 thick."""
        looking_right = (1, 0, 0)
        for x in (-10, -5, -12.5):
            s3d.empty_scene()
            channel = s3d.box("channel", 30, 20, 20)
            s3d.cut(channel, s3d.at(s3d.box("gap", 20, 30, 20), z=5))
            s3d.hole(channel, 3, on=s3d.Frame((x, 0, 12), looking_right), depth=2)
            xs = sorted({round(v.co.x, 4) for v in channel.data.vertices
                         if abs(math.hypot(v.co.y, v.co.z - 12) - 1.5) < 1e-4})
            with self.subTest(frame_x=x):
                self.assertEqual(xs, [-12.0, -10.0], "2 deep into the left wall, from its inner face")

    def test_a_frame_on_the_face_stays_where_it_is(self):
        a = s3d.box("a", 40, 40, 10)
        top = a.face("top")
        s3d.hole(a, 4, on=top.move(5, 5), depth=3)
        b = s3d.box("b", 40, 40, 10)
        s3d.hole(b, 4, on=(5, 5), depth=3)
        self.assertEqual(sorted(tuple(v.co) for v in a.data.vertices), sorted(tuple(v.co) for v in b.data.vertices))


if __name__ == "__main__":
    unittest.main()
