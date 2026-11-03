"""Every solid starts at z = 0, centred in x and y."""
import unittest

import sand3d as s3d
from harness import _object_digest, world
from test_regressions import shells
from test_solids import closed_volume

#: Each maker, called as a program would.
SOLIDS = {"box": lambda: s3d.box("s", 40, 30, 20), "cylinder": lambda: s3d.cylinder("s", 20, 30),
          "tube": lambda: s3d.tube("s", 30, 20, 10), "cone": lambda: s3d.cone("s", 30, 40),
          "frustum": lambda: s3d.cone("s", 30, 40, 10), "sphere": lambda: s3d.sphere("s", 30),
          "wedge": lambda: s3d.wedge("s", 40, 20, 30, 10), "torus": lambda: s3d.torus("s", 40, 6),
          "gear": lambda: s3d.gear("s", 20, 2, 10, bore=8), "dome": lambda: s3d.dome("s", 15, 3),
          "extrude": lambda: s3d.extrude("s", s3d.rect(40, 30), 20),
          "revolve": lambda: s3d.revolve("s", s3d.at(s3d.rect(5, 30), x=12.5, y=15)),
          "thread": lambda: s3d.thread("s", 10, 20, 1.5), "knurl": lambda: s3d.knurl("s", 20, 10)}
HEIGHTS = {"box": 20, "cylinder": 30, "tube": 10, "cone": 40, "frustum": 40, "sphere": 30, "wedge": 30, "torus": 6,
           "gear": 10, "dome": 3, "extrude": 20, "revolve": 30, "thread": 20, "knurl": 10}


def points(ob):
    return [world(ob) @ v.co for v in ob.data.vertices]


class Origin(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_every_solid_starts_at_z_0_centred_in_x_and_y(self):
        for name, make in SOLIDS.items():
            co = points(make())
            with self.subTest(name):
                self.assertEqual((min(c.z for c in co), max(c.z for c in co)), (0, HEIGHTS[name]))
                if name != "thread":                    # a helix is not symmetric about its axis
                    self.assertAlmostEqual(sum(c.x for c in co) / len(co), 0, delta=1e-6)
                    self.assertAlmostEqual(sum(c.y for c in co) / len(co), 0, delta=1e-6)

    def test_a_box_is_the_extrusion_of_its_rectangle(self):
        """To the bit, names included: one origin for both ways of writing it."""
        pairs = [(s3d.box("b", 40, 30, 20), s3d.extrude("b", s3d.rect(40, 30), 20)),
                 (s3d.cylinder("c", 20, 30), s3d.extrude("c", s3d.circle(20), 30)),
                 (s3d.tube("t", 30, 20, 10), s3d.extrude("t", s3d.circle(30) - s3d.circle(20), 10))]
        for solid, extruded in pairs:
            with self.subTest(solid["s3d_name"]):
                self.assertEqual(_object_digest(solid), _object_digest(extruded))

    def test_both_centres_on_z_0(self):
        """`extrude(both=True)` is how a solid is centred on its plane, as box was before 1.0."""
        co = points(s3d.extrude("c", s3d.rect(40, 30), 20, both=True))
        self.assertEqual((min(c.z for c in co), max(c.z for c in co)), (-10, 10))

    def test_a_part_stands_on_a_face_without_anchor(self):
        plate = s3d.box("plate", 60, 40, 8)
        peg = s3d.at(s3d.cylinder("peg", 4, 10), on=plate.face("top"), x=5)
        self.assertEqual(min(c.z for c in points(peg)), 8)
        s3d.join(plate, peg)
        self.assertTrue(closed_volume(plate)[0])
        self.assertEqual(shells(plate), 1)

    def test_a_side_face_is_the_extrusions(self):
        """A face's frame has its origin where the parent's origin projects onto the face:
        on a side face, at z = 0, its foot — as on an extrusion. `.move(y=h / 2)` goes up
        to its middle (the face's Y is the world's Z on a side)."""
        box = s3d.box("b", 40, 30, 20)
        extruded = s3d.extrude("e", s3d.rect(40, 30), 20)
        for side in ("+X", "-X", "+Y", "-Y"):
            with self.subTest(side):
                self.assertEqual(box.face(side).matrix, extruded.face(side).matrix)
                self.assertEqual(box.face(side).origin.z, 0)
                self.assertEqual(box.face(side).move(y=10).origin.z, 10)


if __name__ == "__main__":
    unittest.main()
