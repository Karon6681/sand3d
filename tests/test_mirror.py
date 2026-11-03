"""One meaning for "mirror": the image across a plane."""
import unittest

import sand3d as s3d


class Mirror(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def says(self, error, message, call, *args, **kwargs):
        with self.assertRaises(error) as caught:
            call(*args, **kwargs)
        self.assertEqual(str(caught.exception), message)

    def test_at_has_no_mirror_any_more(self):
        self.says(TypeError, 'at: mirror= is gone in 1.0 — the image across a plane of the parent is '
                             's3d.transform(ob, mirror="x"), on s3d.copy(ob) to keep the original',
                  s3d.at, s3d.box("b", 1, 1, 1), x=2, mirror="x")

    def test_a_profile_mirror_names_the_coordinate_that_changes_sign(self):
        half = s3d.poly([(0, 0), (20, 0), (20, 5), (5, 20), (0, 20)])
        across_y = s3d.mirror(half)                      # x -> -x, as the old axis="y"
        (x0, y0), (x1, y1) = across_y.bounds()
        self.assertEqual((round(x0, 6), round(x1, 6), round(y0, 6), round(y1, 6)), (-20.0, 20.0, 0.0, 20.0))
        self.assertEqual(s3d.mirror(half, "x").mesh()[0], across_y.mesh()[0])
        (x0, y0), (x1, y1) = s3d.mirror(half, "y").bounds()
        self.assertEqual((round(x0, 6), round(x1, 6), round(y0, 6), round(y1, 6)), (0.0, 20.0, -20.0, 20.0))
        self.assertEqual(s3d.mirror(half, "X").mesh()[0], across_y.mesh()[0], "as transform, any case")
        (x0, y0), (x1, y1) = s3d.mirror(s3d.at(half, x=5)).bounds()   # placed in its plane: mirrored there
        self.assertEqual((round(x0, 6), round(x1, 6)), (-25.0, 25.0))

    def test_what_mirror_refuses(self):
        half = s3d.poly([(0, 0), (20, 0), (0, 20)])
        self.says(TypeError, 'mirror: axis= is gone in 1.0 — the letter is the coordinate that changes sign: '
                             's3d.mirror(p, "x") is the old axis="y"', s3d.mirror, half, axis="y")
        for solid in (s3d.box("arm", 4, 4, 4), s3d.assembly("chair", s3d.box("seat", 4, 4, 1))):
            self.says(TypeError, 'mirror: for a solid, s3d.transform(s3d.copy(ob), mirror="x") — its image across '
                                 'the plane x = 0 of its parent', s3d.mirror, solid)
        self.says(ValueError, 'mirror: flip is "x" or "y", the coordinate that changes sign, not \'z\'',
                  s3d.mirror, half, "z")
        self.says(ValueError, "transform: mirror is among 'x', 'y', 'z' — the coordinates that change sign — not 'w'",
                  s3d.transform, s3d.box("b", 1, 1, 1), mirror="w")


if __name__ == "__main__":
    unittest.main()
