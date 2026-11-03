"""cut, hole and pocket raise when no tool cuts into the part."""
import unittest

import bpy

import sand3d as s3d
from test_solids import closed_volume


class NothingCut(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_cut_that_misses_raises_and_leaves_the_part(self):
        plate = s3d.box("plate", 60, 40, 8)
        before = len(plate.data.vertices)
        with self.assertRaises(ValueError) as caught:
            s3d.cut(plate, s3d.at(s3d.cylinder("pin", 10, 20), x=80))
        self.assertEqual(str(caught.exception),
                         "cut: no tool cuts into 'plate' — they miss it, or only touch it (its box (-30, -20, 0) .. "
                         "(30, 20, 8); the tools' (75, -5, 0) .. (85, 5, 20))")
        self.assertEqual(len(plate.data.vertices), before)
        self.assertEqual([o.get("s3d_name") for o in bpy.data.objects], ["plate"], "the tool is used up")

    def test_hole_and_pocket_too(self):
        plate = s3d.box("plate", 60, 40, 8)
        with self.assertRaisesRegex(ValueError, r"^hole: no tool cuts into 'plate' — they miss it, or only touch it "):
            s3d.hole(plate, 5, on=s3d.Frame((100, 0, 0)))
        with self.assertRaisesRegex(ValueError, r"^pocket: no tool cuts into 'plate' — they miss it, or only touch it "):
            s3d.pocket(plate, s3d.circle(8), depth=2, on=s3d.Frame((100, 0, 8)))

    def test_one_tool_among_several_that_misses_is_dropped(self):
        """As in a pattern over an irregular part: the tools that cut stay, the others go."""
        plate = s3d.box("plate", 60, 40, 8)
        s3d.hole(plate, 5, on=[(0, 0), (100, 0)])
        self.assertTrue(closed_volume(plate)[0])
        self.assertLess(closed_volume(plate)[1], 60 * 40 * 8)

    def test_split_keeps_its_silence(self):
        b = s3d.box("b", 20, 20, 20)
        s3d.split(b, s3d.Frame((0, 0, 50)), keep="below")       # the plane misses: nothing to take away
        self.assertAlmostEqual(closed_volume(b)[1], 8000, places=3)

    def test_a_pocket_pattern_drops_a_frame_off_the_part(self):
        plate = s3d.box("plate", 60, 40, 8)
        s3d.pocket(plate, s3d.circle(8), depth=2, on=[s3d.Frame((0, 0, 8)), s3d.Frame((100, 0, 8))])
        self.assertLess(closed_volume(plate)[1], 60 * 40 * 8)
        with self.assertRaisesRegex(ValueError, r"^pocket: no tool cuts into 'plate'"):
            s3d.pocket(plate, s3d.circle(8), depth=2, on=[s3d.Frame((100, 0, 8)), s3d.Frame((-100, 0, 8))])

    def test_a_tool_that_takes_everything_says_so(self):
        """Not "they miss it": nothing would be left."""
        plate = s3d.box("plate", 20, 20, 4)
        with self.assertRaises(ValueError) as caught:
            s3d.cut(plate, s3d.at(s3d.box("big", 200, 200, 200), z=-100))
        self.assertEqual(str(caught.exception), "cut: the tools take all of 'plate' away — nothing would be left")

    def test_the_boxes_are_where_the_program_placed_them(self):
        plate = s3d.at(s3d.box("plate", 60, 40, 8), x=100)
        with self.assertRaises(ValueError) as caught:
            s3d.cut(plate, s3d.at(s3d.cylinder("pin", 10, 20), x=180))
        self.assertIn("(its box (70, -20, 0) .. (130, 20, 8); the tools' (175, -5, 0) .. (185, 5, 20))",
                      str(caught.exception))


if __name__ == "__main__":
    unittest.main()
