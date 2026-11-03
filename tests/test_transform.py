"""transform: from where it is, move it by — as a parent would."""
import unittest

import bpy

import sand3d as s3d
from harness import world


def placed_vertices(ob):
    m = world(ob)
    return sorted(tuple(m @ v.co) for v in ob.data.vertices)


def scene_vertices():
    return sorted(v for ob in bpy.data.objects if ob.type == "MESH" for v in placed_vertices(ob))


class Transform(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_it_adds_to_where_it_is(self):
        b = s3d.at(s3d.box("b", 2, 2, 2), x=10, z=3)
        self.assertIs(s3d.transform(b, x=5), b)
        self.assertEqual(tuple(round(c, 6) for c in world(b).translation), (15.0, 0.0, 3.0))
        s3d.transform(b, x=5, y=-1)
        self.assertEqual(tuple(round(c, 6) for c in world(b).translation), (20.0, -1.0, 3.0))

    def test_it_turns_about_the_parents_origin(self):
        b = s3d.at(s3d.box("b", 2, 2, 2), x=10)
        s3d.transform(b, rz=90)
        self.assertEqual(tuple(round(c, 6) + 0.0 for c in world(b).translation), (0.0, 10.0, 0.0))

    def test_a_door_on_its_hinge_as_with_a_group(self):
        """The door posed itself on its hinge; transform opens it as `at` on a group did."""
        def door():
            return s3d.at(s3d.box("door", 30, 2, 60), x=15, y=-1)
        s3d.at(s3d.group("door", door()), x=1, rz=-30)
        grouped = scene_vertices()
        s3d.empty_scene()
        s3d.transform(door(), x=1, rz=-30)
        self.assertEqual(scene_vertices(), grouped)

    def test_mirror_is_the_image_across_the_parents_plane(self):
        """The arm placed at x = 6 spans 16 .. 46; its image across x = 0 spans -46 .. -16."""
        arm = s3d.extrude("arm", s3d.poly([(10, 0), (40, 0), (40, 20), (30, 20), (10, 6)]), 6)
        s3d.at(arm, x=6, z=2)
        image = s3d.transform(s3d.copy(arm), mirror="x")
        xs = [v[0] for v in placed_vertices(image)]
        self.assertEqual((round(min(xs), 6), round(max(xs), 6)), (-46.0, -16.0))
        self.assertEqual([round(v[0], 6) for v in placed_vertices(arm)][0], 16.0)
        from test_solids import closed_volume
        self.assertTrue(closed_volume(image)[0])

    def test_a_list_a_handle_a_profile_a_path(self):
        tabs = s3d.at(s3d.box("tab", 8, 4, 6), on=s3d.ring(4, 17))
        self.assertEqual(s3d.transform(tabs, z=7), tabs)
        self.assertEqual({round(world(t).translation.z, 6) for t in tabs}, {7.0})
        lamp = s3d.assembly("lamp", s3d.cylinder("stem", 2, 20), top=20)
        self.assertIs(s3d.transform(lamp, x=3), lamp)
        self.assertEqual(round(world(lamp.root).translation.x, 6), 3.0)
        p = s3d.at(s3d.rect(10, 4), x=5)
        q = s3d.transform(p, rz=90)
        self.assertIsNot(q, p)
        self.assertEqual(tuple(round(c, 6) + 0.0 for c in q.matrix.translation), (0.0, 5.0, 0.0))
        self.assertEqual(tuple(round(c, 6) for c in p.matrix.translation), (5.0, 0.0, 0.0))
        path = s3d.transform(s3d.helix(10, 2, 6), z=5)
        self.assertEqual(round(path.points[0][2], 6), 5.0)

    def test_what_it_refuses(self):
        a = s3d.box("a", 2, 2, 2)
        with self.assertRaisesRegex(ValueError, r"^transform: 'a' is passed 2 times; "):
            s3d.transform([a, a], x=5)
        b = s3d.box("b", 2, 2, 2)
        g = s3d.group("g", b)
        with self.assertRaises(ValueError) as caught:
            s3d.transform([g, b], x=5)
        self.assertEqual(str(caught.exception), "transform: 'b' is inside 'g', which this call moves whole: give one "
                                                "of them")
        for frames in (s3d.Frame(), s3d.ring(4, 17)):
            with self.assertRaises(TypeError) as caught:
                s3d.transform(frames, z=7)
            self.assertEqual(str(caught.exception), "transform: a frame moves with its own .move(), .turn() and "
                                                    ".offset(), and so does a list of frames")

    def test_on_gives_a_new_frame_after_a_transform(self):
        """`at(…, on=…)` puts the object in a new frame, keeping the values `at` gave it:
        a transform made before moved the old frame, and is gone — for an object as for
        a profile."""
        a = s3d.transform(s3d.at(s3d.box("a", 2, 2, 2), x=10), z=5, mirror="x")
        copy = s3d.at(a, on=[s3d.Frame((0, 0, 1))])[0]
        self.assertEqual(tuple(round(c, 6) for c in world(copy).translation), (10.0, 0.0, 1.0))
        self.assertGreater(world(copy).determinant(), 0)
        q = s3d.transform(s3d.at(s3d.rect(4, 2), x=10), z=5)
        self.assertEqual(tuple(round(c, 6) for c in s3d.at(q, on=[s3d.Frame()], x=1)[0].matrix.translation),
                         (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
