"""Patterns: `at(ob, on=[frames])`, a list of objects worth its elements, `pocket(on=[frames])`."""
import textwrap
import unittest

import bpy

import sand3d as s3d
from harness import digest, world


def build(src):
    """A program in a fresh scene: the digest of what it built, names included."""
    s3d.empty_scene()
    exec(textwrap.dedent(src).lstrip("\n"), {"s3d": s3d})
    return digest()


def shapes(src):
    """A program in a fresh scene: each object's world vertices and faces, without names."""
    s3d.empty_scene()
    exec(textwrap.dedent(src).lstrip("\n"), {"s3d": s3d})
    out = []
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            m = world(ob)
            out.append((tuple(tuple(m @ v.co) for v in ob.data.vertices), tuple(tuple(p.vertices) for p in ob.data.polygons)))
    return sorted(out)


class TheReportsCases(unittest.TestCase):
    """Each case of the report, the pattern written with `at(on=[frames])`, against the same
    pattern written frame by frame with `copy`."""

    def test_brick(self):
        self.assertEqual(build("""
            brick = s3d.box("brick", 64, 32, 10)
            tool = s3d.at(s3d.cylinder("stud", 9.6, 2), z=10)
            copies = [s3d.at(s3d.copy(tool), on=f) for f in s3d.grid(4, 2, 16, 16)]
            s3d.delete(tool)
            s3d.join(brick, copies)
            s3d.fillet(brick, 0.4, edges="stud")
            """), build("""
            brick = s3d.box("brick", 64, 32, 10)
            studs = s3d.at(s3d.cylinder("stud", 9.6, 2), on=s3d.grid(4, 2, 16, 16), z=10)
            s3d.join(brick, studs)
            s3d.fillet(brick, 0.4, edges="stud")
            """))

    def test_ribs(self):
        self.assertEqual(build("""
            plate = s3d.cylinder("plate", 60, 6)
            tool = s3d.at(s3d.box("rib", 24, 3, 10), z=6)
            copies = [s3d.at(s3d.copy(tool), on=f) for f in s3d.ring(3, 16)]
            s3d.delete(tool)
            s3d.join(plate, copies)
            """), build("""
            plate = s3d.cylinder("plate", 60, 6)
            s3d.join(plate, s3d.at(s3d.box("rib", 24, 3, 10), on=s3d.ring(3, 16), z=6))
            """))

    def test_teeth(self):
        self.assertEqual(build("""
            base = s3d.box("base", 100, 20, 4)
            tool = s3d.at(s3d.box("tooth", 6, 20, 10), z=4)
            copies = [s3d.at(s3d.copy(tool), on=f) for f in s3d.grid(8, 1, 12)]
            s3d.delete(tool)
            s3d.join(base, copies)
            """), build("""
            base = s3d.box("base", 100, 20, 4)
            s3d.join(base, s3d.at(s3d.box("tooth", 6, 20, 10), on=s3d.grid(8, 1, 12), z=4))
            """))

    def test_beads(self):
        """A pattern that is not fused: the list goes to `group` as it is. The loop had to
        name each bead, so only the shapes compare."""
        self.assertEqual(shapes("""
            path = s3d.spline([(0, 0, 0), (30, 20, 0), (60, 0, 0), (90, 20, 0)])
            rail = s3d.wire("rail", path, 4)
            beads = []
            for k, f in enumerate(s3d.along(path, count=6)):
                beads.append(s3d.at(s3d.tube(f"bead{k}", 12, 4, 3), on=f))
            s3d.group("necklace", rail, *beads)
            """), shapes("""
            path = s3d.spline([(0, 0, 0), (30, 20, 0), (60, 0, 0), (90, 20, 0)])
            rail = s3d.wire("rail", path, 4)
            beads = s3d.at(s3d.tube("bead", 12, 4, 3), on=s3d.along(path, count=6))
            s3d.group("necklace", rail, beads)
            """))

    def test_chairs(self):
        """A part already built (seat, legs, back) repeated round a table."""
        self.assertEqual(build("""
            def chair():
                seat = s3d.box("seat", 40, 40, 4)
                tool = s3d.at(s3d.cylinder("leg", 4, 40), z=-40)
                copies = [s3d.at(s3d.copy(tool), on=f) for f in s3d.grid(2, 2, 32, 32)]
                s3d.delete(tool)
                s3d.join(seat, copies)
                s3d.join(seat, s3d.at(s3d.box("back", 40, 4, 40), y=-18, z=4))
                return seat
            table = s3d.at(s3d.cylinder("table", 100, 4), z=70)
            s3d.join(table, s3d.cylinder("foot", 8, 70))
            for f in s3d.ring(4, 90):
                s3d.at(chair(), on=f, z=40, rz=90)
            """), build("""
            seat = s3d.box("seat", 40, 40, 4)
            s3d.join(seat, s3d.at(s3d.cylinder("leg", 4, 40), on=s3d.grid(2, 2, 32, 32), z=-40))
            s3d.join(seat, s3d.at(s3d.box("back", 40, 4, 40), y=-18, z=4))
            table = s3d.at(s3d.cylinder("table", 100, 4), z=70)
            s3d.join(table, s3d.cylinder("foot", 8, 70))
            s3d.at(seat, on=s3d.ring(4, 90), z=40, rz=90)
            """))

    def test_pockets(self):
        self.assertEqual(build("""
            plate = s3d.box("plate", 80, 30, 8)
            for f in s3d.grid(3, 1, 20, on=plate.face("top")):
                s3d.pocket(plate, s3d.circle(8), depth=3, on=f)
            """), build("""
            plate = s3d.box("plate", 80, 30, 8)
            s3d.pocket(plate, s3d.circle(8), depth=3, on=s3d.grid(3, 1, 20, on=plate.face("top")))
            """))


class OnFrames(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_the_object_is_used_up_and_its_copies_are_named_like_it(self):
        b = s3d.box("b", 10, 10, 10)
        copies = s3d.at(b, on=s3d.ring(4, 30))
        self.assertEqual(len(copies), 4)
        self.assertEqual([c["s3d_name"] for c in copies], ["b"] * 4)
        self.assertEqual([tuple(round(v, 6) + 0.0 for v in world(c).translation) for c in copies],
                         [(30.0, 0.0, 0.0), (0.0, 30.0, 0.0), (-30.0, 0.0, 0.0), (0.0, -30.0, 0.0)])
        self.assertEqual(len(bpy.data.objects), 4, "no copy is left at the origin")
        with self.assertRaises(ValueError) as caught:
            s3d.at(b, x=1)
        self.assertEqual(str(caught.exception),
                         "at: 'b' was used up by at(b, ...) on line "
                         + str(caught.exception).split("on line ")[1].split(";")[0]
                         + "; at with a list of frames uses the object up and returns one copy per frame: "
                           "use the list it returns.")

    def test_the_type_of_on_decides(self):
        b = s3d.box("b", 10, 10, 10)
        self.assertIs(s3d.at(b, on=s3d.Frame((5, 0, 0))), b)
        one = s3d.at(b, on=[s3d.Frame((5, 0, 0))])
        self.assertIsInstance(one, list)
        self.assertEqual(len(one), 1)

    def test_the_objects_placement_holds_in_each_frame(self):
        """The slot stays 4 up in every frame of the ring, as join(at=) kept it; values
        given with the frames replace the ones they name."""
        slot = s3d.at(s3d.box("slot", 6, 50, 4), z=4, rz=10)
        slots = s3d.at(slot, on=s3d.ring(3, 0))
        for c in slots:
            self.assertAlmostEqual(world(c).translation.z, 4, places=6)
        raised = s3d.at(s3d.at(s3d.box("tab", 8, 4, 6), x=17, z=2), on=[s3d.Frame(), s3d.Frame((0, 0, 10))], z=7)
        self.assertEqual([tuple(round(v, 6) + 0.0 for v in world(c).translation) for c in raised],
                         [(17.0, 0.0, 7.0), (17.0, 0.0, 17.0)])

    def test_anchor_applies_per_copy(self):
        posts = s3d.at(s3d.cylinder("post", 4, 10), on=s3d.grid(2, 1, 20), anchor="top")
        for c in posts:
            self.assertAlmostEqual(max((world(c) @ v.co).z for v in c.data.vertices), 0, places=6)

    def test_a_group_and_an_assembly_are_copied_whole(self):
        g = s3d.group("pair", s3d.box("a", 4, 4, 4), s3d.at(s3d.box("b", 4, 4, 4), x=6))
        pairs = s3d.at(g, on=s3d.grid(1, 3, 0, 20))
        self.assertEqual([len(p.children) for p in pairs], [2, 2, 2])
        lamp = s3d.assembly("lamp", s3d.cylinder("stem", 2, 20), top=20)
        lamps = s3d.at(lamp, on=s3d.ring(3, 50))
        self.assertTrue(all(isinstance(h, s3d.Handle) and h.top == 20 for h in lamps))
        self.assertEqual(sum(1 for o in bpy.data.objects if o.get("s3d_name") == "stem"), 3)

    def test_a_profile_gives_a_list_of_placed_profiles(self):
        holes = s3d.at(s3d.circle(6), on=s3d.grid(3, 1, 20))
        self.assertEqual(len(holes), 3)
        plate = s3d.rect(80, 30) - holes[0] - holes[1] - holes[2]
        self.assertEqual(len(plate.loops), 4)

    def test_a_copy_keeps_the_anchor_point_it_was_placed_by(self):
        """Anchored on its top, then grown above: each copy sits where `slot` sat in its
        own frame, not re-anchored on the grown box."""
        slot = s3d.at(s3d.box("slot", 6, 50, 4), anchor="top")
        s3d.join(slot, s3d.at(s3d.box("lip", 6, 50, 3), z=-1.5))
        z = world(slot).translation.z
        for c in s3d.at(slot, on=s3d.ring(3, 0)):
            self.assertAlmostEqual(world(c).translation.z, z, places=6)

    def test_a_profile_given_a_value_twice_takes_the_last(self):
        p = s3d.at(s3d.circle(4), z=4, x=3)
        q = s3d.at(p, on=[s3d.Frame()], z=7)[0]
        self.assertEqual(tuple(round(v, 6) for v in q.matrix.translation), (3.0, 0.0, 7.0))

    def test_a_list_given_twice_or_with_a_part_of_another(self):
        b = s3d.box("b", 1, 1, 1)
        with self.assertRaisesRegex(ValueError, r"^at: 'b' is passed 2 times; "):
            s3d.at([b, b], on=s3d.ring(3, 10))
        a = s3d.box("a", 1, 1, 1)
        g = s3d.group("g", a)
        with self.assertRaises(ValueError) as caught:
            s3d.at([g, a], on=s3d.ring(3, 10))
        self.assertEqual(str(caught.exception), "at: 'a' is inside 'g', which this call moves whole: give one of them")

    def test_nothing_is_made_when_it_raises(self):
        b = s3d.box("b", 1, 1, 1)
        with self.assertRaisesRegex(ValueError, r"'bootom'"):
            s3d.at(b, on=s3d.ring(4, 10), anchor="bootom")
        self.assertEqual(len(bpy.data.objects), 1)
        s3d.at(b, x=1)

    def test_the_parts_of_a_used_up_assembly_say_so(self):
        stem = s3d.cylinder("stem", 2, 20)
        lamp = s3d.assembly("lamp", stem)
        s3d.at(lamp, on=s3d.ring(3, 50))
        with self.assertRaisesRegex(ValueError, r"^fillet: 'stem' was used up by at\(lamp, \.\.\.\) on line \d+; "
                                                r"at with a list of frames uses the object up"):
            s3d.fillet(stem, 0.5)

    def test_on_must_be_frames(self):
        with self.assertRaisesRegex(TypeError, r"^at: on is a frame or a list of frames"):
            s3d.at(s3d.box("b", 1, 1, 1), on=[(0, 0, 0)])


class ListsOfObjects(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_at_acts_on_each(self):
        hub = s3d.cylinder("hub", 30, 20)
        tabs = s3d.at(s3d.box("tab", 8, 4, 6), on=s3d.ring(4, 17))
        self.assertEqual(s3d.at(tabs, z=7), tabs)
        self.assertEqual([round(world(t).translation.z, 6) for t in tabs], [7.0] * 4)
        s3d.join(hub, tabs)
        self.assertEqual(len(bpy.data.objects), 1)

    def test_nested_lists_and_lists_of_placed_profiles(self):
        studs = s3d.at(s3d.cylinder("stud", 4, 2), on=s3d.grid(2, 1, 10))
        ribs = s3d.at(s3d.box("rib", 2, 8, 2), on=s3d.grid(1, 2, 0, 10))
        s3d.at([studs, ribs], z=5)
        self.assertEqual({round(world(o).translation.z, 6) for o in studs + ribs}, {5.0})
        holes = s3d.at(s3d.at(s3d.circle(6), on=s3d.grid(3, 1, 20)), y=4)
        self.assertEqual([round(h.matrix.translation.y, 6) for h in holes], [4.0] * 3)

    def test_a_list_of_points_stays_a_path(self):
        path = s3d.at([(0, 0, 0), (10, 0, 0)], z=5)
        self.assertEqual(path.points, [(0.0, 0.0, 5.0), (10.0, 0.0, 5.0)])

    def test_anchor_on_a_list_is_refused(self):
        tabs = s3d.at(s3d.box("tab", 8, 4, 6), on=s3d.ring(2, 17))
        with self.assertRaisesRegex(ValueError, r"^at: anchor on a list of objects"):
            s3d.at(tabs, anchor="bottom")

    def test_every_objects_parameter_takes_lists(self):
        def parts(n, name="p"):
            return s3d.at(s3d.box(name, 2, 2, 2), on=s3d.grid(n, 1, 10))
        plate = s3d.box("plate", 100, 20, 2)
        s3d.join(plate, parts(3), [s3d.at(s3d.box("q", 2, 2, 2), z=2)])
        s3d.cut(plate, parts(2, "notch"))
        m = s3d.merge(parts(2, "m"))
        self.assertEqual(m["s3d_name"], "m")
        g = s3d.group("g", parts(2), [m])
        self.assertEqual(len(g.children), 3)
        h = s3d.assembly("a", parts(2))
        self.assertEqual(len(h.root.children), 2)
        d = s3d.attach(h, "detail", parts(3))
        self.assertEqual(len(d.children), 3)
        self.assertEqual(len(s3d.group("of handles", s3d.at(h, on=s3d.ring(2, 40))).children), 2)


if __name__ == "__main__":
    unittest.main()
