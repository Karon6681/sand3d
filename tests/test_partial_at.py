"""at sets the values it is given and keeps the others; on= changes the frame."""
import unittest

import sand3d as s3d
from harness import world


def where(ob):
    return tuple(round(c, 6) + 0.0 for c in world(ob).translation)


class PartialAt(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_the_values_not_given_are_kept(self):
        """The report's groove: up to the top, then 15 along. Today the second at put it back at z = 0."""
        groove = s3d.box("groove", 8, 44, 6)
        s3d.at(groove, z=5)
        s3d.at(groove, x=15)
        self.assertEqual(where(groove), (15.0, 0.0, 5.0))
        s3d.at(groove, rz=90)
        self.assertEqual(where(groove), (15.0, 0.0, 5.0))
        s3d.at(groove, x=0)
        self.assertEqual(where(groove), (0.0, 0.0, 5.0))

    def test_a_single_call_is_as_before(self):
        """The first `at` on a solid sets every value, the ones not given at 0."""
        a = s3d.at(s3d.box("a", 10, 20, 30), x=5, y=-3, z=2, rx=10, ry=20, rz=30)
        b = s3d.box("b", 10, 20, 30)
        b.matrix_basis = __import__("mathutils").Matrix.Translation((5, -3, 2)) @ __import__("mathutils").Euler(
            tuple(__import__("math").radians(v) for v in (10, 20, 30))).to_matrix().to_4x4()
        self.assertEqual(tuple(map(tuple, a.matrix_basis)), tuple(map(tuple, b.matrix_basis)))

    def test_on_changes_the_frame_and_keeps_the_values(self):
        slot = s3d.at(s3d.box("slot", 6, 50, 4), z=4)
        s3d.at(slot, on=s3d.Frame((10, 0, 0)))
        self.assertEqual(where(slot), (10.0, 0.0, 4.0))
        s3d.at(slot, x=3)                      # in the frame it is in now
        self.assertEqual(where(slot), (13.0, 0.0, 4.0))

    def test_at_with_nothing_changes_nothing(self):
        b = s3d.at(s3d.box("b", 2, 2, 2), x=4, rz=30)
        before = tuple(map(tuple, b.matrix_basis))
        s3d.at(b)
        self.assertEqual(tuple(map(tuple, b.matrix_basis)), before)
        fresh = s3d.extrude("e", s3d.rect(4, 4), 2, on=s3d.Frame((0, 0, 7), (1, 0, 0)))
        before = tuple(map(tuple, fresh.matrix_basis))
        s3d.at(fresh)
        self.assertEqual(tuple(map(tuple, fresh.matrix_basis)), before)

    def test_a_part_built_on_a_face_is_placed_in_that_face(self):
        """A verb that sets its part on a frame gives it that frame: at moves it within."""
        plate = s3d.box("plate", 60, 40, 8)
        boss = s3d.extrude("boss", s3d.circle(10), 5, on=plate.face("+X"))
        s3d.at(boss, x=5)
        self.assertEqual(where(boss), (30.0, 5.0, 0.0))   # on +X the face's X is world +Y

    def test_the_anchor_is_kept(self):
        b = s3d.at(s3d.box("b", 10, 10, 10), anchor="top")
        s3d.at(b, x=20)
        self.assertEqual(where(b), (20.0, 0.0, -10.0))

    def test_a_transform_moves_the_frame_at_works_in(self):
        """`at` after a `transform` places the part in the moved frame, as in a group."""
        door = s3d.at(s3d.box("door", 30, 2, 60), x=15, y=-1)
        s3d.transform(door, rz=90)
        s3d.at(door, z=1)
        self.assertEqual(tuple(round(c, 5) + 0.0 for c in where(door)), (1.0, 15.0, 1.0))

    def test_a_handle_by_its_root(self):
        lamp = s3d.assembly("lamp", s3d.cylinder("stem", 2, 20))
        s3d.at(lamp, x=10)
        s3d.at(lamp, y=5)
        self.assertEqual(where(lamp.root), (10.0, 5.0, 0.0))

    def test_profiles_and_paths_follow_the_same_rule(self):
        """Today their placements added up: at(p, z=4) then at(·, z=7) gave z = 11."""
        p = s3d.at(s3d.at(s3d.circle(4), z=4, x=3), z=7)
        self.assertEqual(tuple(round(c, 6) for c in p.matrix.translation), (3.0, 0.0, 7.0))
        once = s3d.at(s3d.rect(10, 5), x=3, rz=30)
        self.assertEqual(tuple(round(c, 6) for c in once.matrix.translation), (3.0, 0.0, 0.0))
        path = s3d.at(s3d.at(s3d.helix(10, 2, 6), z=5), x=2)
        self.assertEqual(tuple(round(c, 6) for c in path.points[0]), (7.0, 0.0, 5.0))

    def test_a_move_bpy_made_goes_to_the_frame(self):
        """An object moved by raw bpy since its last `at`: the move goes to its frame, and
        the values, the mirror and the anchor `at` set stay — none is applied twice."""
        b = s3d.at(s3d.box("b", 2, 2, 2), x=4)
        b.location.z = 9                        # raw
        s3d.at(b, y=1)
        self.assertEqual(where(b), (4.0, 1.0, 9.0))
        c = s3d.transform(s3d.at(s3d.box("c", 10, 10, 10), anchor="top", rz=30), mirror="y")
        c.location.x += 3                       # raw
        s3d.at(c, anchor="top")
        self.assertLess(world(c).determinant(), 0, "still mirrored once")
        self.assertAlmostEqual(max((world(c) @ v.co).z for v in c.data.vertices), 0, places=5)
        s3d.at(c, rz=90)                        # set to 90, not 30 + 90, in the mirrored frame: X along -Y
        self.assertAlmostEqual(world(c).col[0].xyz.y, -1, places=5)
        self.assertLess(world(c).determinant(), 0)
        self.assertAlmostEqual(world(c).translation.x, 3, places=5)

    def test_export_leaves_the_model_in_its_unit(self):
        import tempfile
        from pathlib import Path
        root = s3d.group("thing", s3d.box("b", 10, 10, 10))
        with tempfile.TemporaryDirectory() as tmp:
            s3d.export(root, Path(tmp) / "thing.glb")
        self.assertEqual(tuple(root.scale), (1.0, 1.0, 1.0))
        s3d.at(root, x=5)
        self.assertEqual(where(root), (5.0, 0.0, 0.0))

    def test_programs_that_placed_twice(self):
        """at(slot_tool, z=-c), then at(slot_tool, y=Y); or a slot at z=-1
        then repeated on a ring — the height is kept in both."""
        tool = s3d.box("slot_tool", 4, 4, 4)
        s3d.at(tool, z=-2)
        s3d.at(tool, y=12)
        self.assertEqual(where(tool), (0.0, 12.0, -2.0))
        slot = s3d.at(s3d.box("slot", 2, 30, 2), z=-1)
        self.assertEqual({where(c)[2] for c in s3d.at(slot, on=s3d.ring(4, 0))}, {-1.0})


if __name__ == "__main__":
    unittest.main()
