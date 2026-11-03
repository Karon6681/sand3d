"""`local`: a frame read on a part at a point of its own space — a source's joint
origin, xyz and rpy, as written — used both to place the child and to hang it."""
import math
import unittest

import bpy  # noqa: F401
from mathutils import Matrix, Vector

import sand3d as s3d
from sand3d import _core, joints as k


def close(test, a, b, places=6):
    for x, y in zip(a, b):
        test.assertAlmostEqual(x, y, places=places)


def rpy(rx, ry, rz):
    """URDF's rotation of an origin's rpy: about the fixed X, then Y, then Z."""
    return (Matrix.Rotation(math.radians(rz), 3, "Z") @ Matrix.Rotation(math.radians(ry), 3, "Y")
            @ Matrix.Rotation(math.radians(rx), 3, "X"))


class Local(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_point_of_its_own_space_follows_where_the_part_was_placed(self):
        bar = s3d.at(s3d.box("bar", 100, 10, 10), x=100, z=5, rz=90)
        f = s3d.local(bar, x=40, z=10)
        close(self, f.origin, (100, 40, 15))      # its own X runs along +Y once it is turned
        close(self, f.x, (0, 1, 0))
        close(self, f.z, (0, 0, 1))
        close(self, s3d.local(bar).origin, (100, 0, 5))

    def test_turned_as_at_turns_which_is_urdf_s_rpy(self):
        cab = s3d.box("cab", 600, 600, 850)
        f = s3d.local(cab, rx=30, ry=40, rz=50)
        m = rpy(30, 40, 50)
        close(self, f.x, m.col[0])
        close(self, f.y, m.col[1])
        close(self, f.z, m.col[2])
        probe = s3d.at(s3d.box("probe", 1, 1, 1), x=5, rx=30, ry=40, rz=50)
        close(self, probe.matrix_basis.to_3x3().col[2], f.z)

    def test_where_the_part_is_built_its_top_lies_on_its_top_face(self):
        cab = s3d.at(s3d.box("cab", 600, 600, 850), x=40, rz=30)
        top, face = s3d.local(cab, z=850), cab.face("top")
        close(self, top.origin, (40, 0, 850))       # the face's frame starts where the parent's origin projects
        self.assertAlmostEqual(top.origin.z, face.origin.z)
        close(self, top.z, face.z)
        close(self, s3d.local(s3d.box("crate", 200, 200, 300), z=300).origin, (0, 0, 300))

    def test_in_the_space_the_part_lives_in_as_face_and_bbox_are(self):
        cab = s3d.at(s3d.box("cab", 600, 600, 850), x=300)
        unit = s3d.transform(s3d.group("unit", cab), rz=90, x=1000)
        close(self, s3d.local(cab, z=850).origin, (300, 0, 850))      # in the group's space
        close(self, s3d.local(unit, x=300, z=850).origin, (1000, 300, 850))

    def test_it_is_read_on_the_part_through_move_turn_and_offset(self):
        cab = s3d.box("cab", 600, 600, 850)
        f = s3d.local(cab, x=300, z=400)
        self.assertIs(f.part, cab)
        self.assertIs(f.move(y=10).turn(rz=30).offset(5).part, cab)
        handle = s3d.assembly("set", cab)
        self.assertIs(s3d.local(handle, z=10).part, handle.root)

    def test_one_frame_places_the_child_and_hangs_it_on_a_turned_parent(self):
        base = s3d.cylinder("base", 160, 20)
        shoulder = s3d.local(base, z=20, ry=-30)
        lower = s3d.at(s3d.group("lower", s3d.at(s3d.box("beam", 300, 30, 30), x=150)), on=shoulder)
        s3d.joint("shoulder", lower, shoulder, axis="+Y", range=(-45, 90))
        elbow = s3d.local(lower, x=300, rx=90, rz=15)
        upper = s3d.at(s3d.group("upper", s3d.at(s3d.box("rod", 200, 20, 20), x=100)), on=elbow)
        s3d.joint("elbow", upper, elbow, axis="normal", range=(-90, 90))
        pin = _core.world(lower) @ Vector((300, 0, 0))
        p, a = k.axis_in_world(upper, 0)
        close(self, p, pin, 4)
        close(self, _core.world(upper).to_translation(), pin, 4)       # the child sits on its pivot
        close(self, a, _core.world(lower).to_3x3() @ rpy(90, 0, 15).col[2])
        self.assertIs(k.parent_of(upper), lower)

    def test_the_pivot_rides_with_its_parent(self):
        base = s3d.cylinder("base", 160, 20)
        shoulder = s3d.local(base, z=20)
        lower = s3d.at(s3d.group("lower", s3d.at(s3d.box("beam", 300, 30, 30), x=150)), on=shoulder)
        s3d.joint("shoulder", lower, shoulder, axis="+Y", range=(-45, 90))
        elbow = s3d.local(lower, x=300, z=15)
        upper = s3d.at(s3d.box("rod", 200, 20, 20), on=elbow)
        s3d.joint("elbow", upper, elbow, axis="+Y", range=(-90, 90))
        s3d.transform(s3d.group("arm", base, lower, upper), x=500, rz=40)
        s3d.pose(lower, 60)
        s3d.pose(upper, -30)
        close(self, k.axis_in_world(upper, 0)[0], _core.world(lower) @ Vector((300, 0, 15)), 4)

    def test_a_length_is_in_the_program_s_unit(self):
        s3d.units("cm")
        cab = s3d.box("cab", 60, 60, 85)
        close(self, s3d.local(cab, x=30, z=85).origin, (30, 0, 85))

    def test_what_is_no_part_or_no_number_is_refused(self):
        for bad in (s3d.Frame(), (0, 0, 400), [s3d.box("a", 1, 1, 1)]):
            with self.subTest(bad=type(bad).__name__), self.assertRaisesRegex(
                    TypeError, rf"^local: a part, a group or an assembly is expected, not {type(bad).__name__}: "
                               r"the frame is read on it, at a point of its own space$"):
                s3d.local(bad, x=400)
        cab = s3d.box("cab", 600, 600, 850)
        for key, value in (("x", "400"), ("rz", True), ("y", None)):
            with self.subTest(key=key), self.assertRaisesRegex(TypeError, rf"^local: {key} is a number, not "):
                s3d.local(cab, **{key: value})


if __name__ == "__main__":
    unittest.main()
