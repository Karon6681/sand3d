"""Frames read on parts: `bbox`, the part a frame remembers, and
`at`'s anchor by directions."""
import unittest

import bpy  # noqa: F401
from mathutils import Matrix

import sand3d as s3d


def close(test, a, b, places=6):
    for x, y in zip(a, b):
        test.assertAlmostEqual(x, y, places=places)


class Bbox(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab = s3d.box("cab", 600, 600, 850)

    def test_a_side_is_the_centre_of_that_side_with_a_faces_axes(self):
        f = s3d.bbox(self.cab, "+X")
        close(self, f.origin, (300, 0, 425))
        close(self, f.z, (1, 0, 0))
        close(self, f.x, (0, 1, 0))             # on a face looking +X, X is +Y and Y is up
        close(self, f.y, (0, 0, 1))
        top = s3d.bbox(self.cab, "+Z")
        close(self, top.origin, (0, 0, 850))
        close(self, top.x, (1, 0, 0))

    def test_an_edge_and_a_corner_take_the_axes_of_the_first_side_named(self):
        edge = s3d.bbox(self.cab, "+X-Y")
        close(self, edge.origin, (300, -300, 425))
        close(self, edge.z, (1, 0, 0))
        other = s3d.bbox(self.cab, "-Y+X")
        close(self, other.origin, (300, -300, 425))
        close(self, other.z, (0, -1, 0))
        corner = s3d.bbox(self.cab, "+X-Y+Z")
        close(self, corner.origin, (300, -300, 850))
        close(self, s3d.bbox(self.cab, "Y").origin, (0, 300, 425))

    def test_a_face_frame_starts_at_the_projected_origin_a_bbox_side_at_its_centre(self):
        close(self, self.cab.face("+X").origin, (300, 0, 0))
        close(self, s3d.bbox(self.cab, "+X").origin, (300, 0, 425))

    def test_a_group_s_box_holds_everything_under_it_in_the_space_it_is_in(self):
        knob = s3d.at(s3d.box("knob", 20, 20, 20), x=310, z=400)
        unit = s3d.group("unit", self.cab, knob)
        close(self, s3d.bbox(unit, "+X").origin, (320, 0, 425))
        s3d.transform(unit, rz=90)                       # its box, along its parent's axes
        close(self, s3d.bbox(unit, "+Y").origin, (0, 320, 425))
        close(self, s3d.bbox(unit, "+Y").z, (0, 1, 0))
        close(self, s3d.bbox(self.cab, "+X").origin, (300, 0, 425))     # the cab, in the group's space
        handle = s3d.assembly("set", unit)
        close(self, s3d.bbox(handle, "+Y").origin, (0, 320, 425))

    def test_a_turned_part_s_box_is_along_its_parent_s_axes(self):
        bar = s3d.at(s3d.box("bar", 100, 10, 10), rz=90)
        close(self, s3d.bbox(bar, "+Y").origin, (0, 50, 5))

    def test_what_names_no_side_is_refused(self):
        message = (r"^bbox: '{}' names no side of a box: a side \"\+X\", an edge \"\+X-Y\" or a corner \"\+X-Y\+Z\" — "
                   r"a sign and an axis for each, every axis once$")
        for bad in ("+X+X", "XY", "+Q", "+X-Y+Z-X", "", "top", "+x-"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message.format(bad.replace("+", r"\+"))):
                s3d.bbox(self.cab, bad)

    def test_a_group_with_no_solid_has_no_box(self):
        with self.assertRaisesRegex(ValueError, r"^bbox: 'void' holds no solid: it has no box$"):
            s3d.bbox(s3d.group("void"), "+X")
        with self.assertRaisesRegex(TypeError, r"^bbox: a part, a group or an assembly is expected, not Frame$"):
            s3d.bbox(s3d.Frame(), "+X")


class PartFrames(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab = s3d.box("cab", 600, 600, 850)

    def test_a_frame_read_on_a_part_remembers_it_through_move_turn_offset(self):
        for f in (self.cab.face("+X"), s3d.bbox(self.cab, "+X-Y")):
            for g in (f, f.move(y=10), f.turn(rz=30), f.offset(5), f.move(1, 2, 3).turn(rx=5).offset(-2)):
                self.assertIs(g.part, self.cab)
            for g in s3d.grid(2, 2, 10, 10, on=f) + s3d.ring(3, 10, on=f):
                self.assertIs(g.part, self.cab)
        self.assertIsNone(s3d.Frame().part)
        self.assertIsNone(s3d.Frame().move(1).part)

    def test_face_builds_what_it_built(self):
        f = self.cab.face("+X")
        self.assertEqual(repr(f), "Frame(origin=(300.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))")
        self.assertEqual(f.matrix, s3d.Frame((300, 0, 0), (1, 0, 0)).matrix)

    def test_the_frame_stays_on_its_part_when_the_part_moves(self):
        f = self.cab.face("+X").move(x=-300)               # its X is +Y: the -Y edge's foot
        close(self, f.origin, (300, -300, 0))
        s3d.transform(self.cab, x=100, rz=90)
        from sand3d import _core
        here = _core.world(self.cab) @ f._in_part()
        close(self, here.translation, (400, 300, 0))
        close(self, here.col[2].xyz, (0, 1, 0))

    def test_an_assembly_s_frame_anchor_is_read_on_the_assembly(self):
        body = s3d.box("body", 10, 10, 100)
        arm = s3d.assembly("arm", body, tip=s3d.Frame((0, 0, 100)), top=body.face("top"), size=100,
                           holes=s3d.grid(2, 1, 5))
        self.assertIs(arm.tip.part, arm.root)
        self.assertEqual(arm.tip._basis, Matrix.Identity(4))
        self.assertIs(arm.top.part, body)                 # read on a part already: it keeps it
        self.assertEqual([f.part for f in arm.holes], [arm.root, arm.root])
        self.assertIsInstance(arm.holes, type(s3d.grid(1)))
        self.assertEqual(arm.size, 100)
        twin = s3d.copy(arm)
        self.assertIs(twin.tip.part, twin.root)
        self.assertIsNot(twin.top.part, body)
        self.assertEqual(twin.top.part.get("s3d_name"), "body")


class AnchorDirections(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_directions_name_the_same_points_as_words(self):
        for words, directions in (("bottom", "-Z"), ("top-front", "+Z-Y"), ("bottom-left-back", "-Z-X+Y"),
                                  ("right", "+X"), ("left", "X")):
            a = s3d.at(s3d.box("a", 10, 20, 30), x=1, anchor=words)
            b = s3d.at(s3d.box("b", 10, 20, 30), x=1, anchor=directions)
            with self.subTest(words=words):
                if directions == "X":
                    close(self, a.location, (6, 0, -15))
                    close(self, b.location, (-4, 0, -15))
                else:
                    close(self, a.location, b.location)

    def test_a_wrong_anchor_says_both_vocabularies(self):
        with self.assertRaisesRegex(ValueError, r"^anchor 'bootom': 'bootom' is none of \[.*\], and the anchor is no "
                                                r"directions either \(\"-Z\", \"\+Y-Z\", \"\+X-Y\+Z\"\)$"):
            s3d.at(s3d.box("a", 1, 1, 1), anchor="bootom")
        with self.assertRaisesRegex(ValueError, r"^anchor '\+Z\+Z': '\+Z\+Z' names no side of a box"):
            s3d.at(s3d.box("b", 1, 1, 1), anchor="+Z+Z")


if __name__ == "__main__":
    unittest.main()
