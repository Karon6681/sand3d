"""extrude and pocket take a profile drawn in its plane; the face goes in on=."""
import unittest

import sand3d as s3d
from test_solids import closed_volume

EXTRUDE = ('extrude: the profile is placed out of its plane (moved in z, or turned off it): draw it in its plane — '
           'x, y and rz only — and give the face or frame with on=: s3d.extrude(name, profile, h, on=plate.face("top"))')
POCKET = ("pocket: the profile is placed out of the face's plane: draw it in its plane — x, y and rz only — and give "
          "the face or frames with on=: s3d.pocket(ob, profile, depth, on=…)")


class InPlane(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def says(self, message, call, *args, **kwargs):
        with self.assertRaises(ValueError) as caught:
            call(*args, **kwargs)
        self.assertEqual(str(caught.exception), message)

    def test_a_drawing_moved_within_its_plane_still_works(self):
        plate = s3d.rect(40, 20) - s3d.at(s3d.circle(6), x=12)
        s3d.extrude("plate", plate, 4)
        boss = s3d.extrude("boss", s3d.at(s3d.rect(10, 4), x=5, y=-3, rz=30), 6)
        self.assertTrue(closed_volume(boss)[0])
        s3d.extrude("image", s3d.transform(s3d.rect(10, 4), mirror="x"), 2)
        base = s3d.box("base", 40, 40, 10)
        s3d.pocket(base, s3d.at(s3d.circle(8), x=10), depth=3)
        moved = s3d.extrude("rib", s3d.at(s3d.rect(4, 4), x=10, rz=30), until=s3d.Frame((0, 0, 20)))
        drawn = s3d.extrude("rib", s3d.at(s3d.rect(4, 4), x=10, rz=30), 20)
        self.assertEqual(tuple(map(tuple, moved.matrix_basis)), tuple(map(tuple, drawn.matrix_basis)),
                         "with until as with h, a move within the plane is part of the drawing")

    def test_a_drawing_placed_off_its_plane_is_refused(self):
        plate = s3d.box("plate", 60, 40, 8)
        for placed in (s3d.at(s3d.circle(10), z=3), s3d.at(s3d.circle(10), rx=90),
                       s3d.at(s3d.circle(10), on=plate.face("top"))):
            with self.subTest(placed=placed.matrix.translation[:]):
                self.says(EXTRUDE, s3d.extrude, "boss", placed, 5)
        self.says(EXTRUDE, s3d.extrude, "rib", s3d.at(s3d.rect(4, 4), z=2), until=s3d.Frame((0, 0, 20)))
        self.says(EXTRUDE, s3d.extrude, "tilt", s3d.at(s3d.rect(200, 100), rx=0.05), 5)   # corners 0.09 mm off
        base = s3d.box("base", 40, 40, 10)
        for placed in (s3d.at(s3d.circle(8), rx=90), s3d.at(s3d.circle(8), on=base.face("top"))):
            self.says(POCKET, s3d.pocket, base, placed, depth=2)

    def test_the_pocket_trap_of_the_report(self):
        """A grid set on the top places the circles 4 up: pocket read them in the face's plane
        and cut one pocket at the centre, without a word. The frames go in on=."""
        plate = s3d.box("plate", 80, 30, 8)
        top = plate.face("top")
        circles = s3d.at(s3d.circle(8), on=s3d.grid(3, 1, 20, on=top))
        self.says(POCKET, s3d.pocket, plate, circles[1], depth=3)
        s3d.pocket(plate, s3d.circle(8), depth=3, on=s3d.grid(3, 1, 20, on=top))
        self.assertAlmostEqual(closed_volume(plate)[1], 80 * 30 * 8 - 3 * 3 * 3.1416 * 16, delta=12)

    def test_loft_sections_are_placed_in_space_as_before(self):
        vase = s3d.loft("vase", s3d.circle(40), s3d.at(s3d.circle(60), z=50), s3d.at(s3d.circle(30), z=100, rx=10))
        self.assertTrue(closed_volume(vase)[0])


if __name__ == "__main__":
    unittest.main()
