"""What a program does wrong with an object, said by the next verb that meets it."""
import textwrap
import unittest

import bpy

import sand3d as s3d


def run(src):
    """Run a small program, as a file of its own: line numbers are the program's."""
    exec(compile(textwrap.dedent(src).lstrip("\n"), "program.py", "exec"), {"s3d": s3d})


class UsedUp(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def message(self, src):
        with self.assertRaises(ValueError) as caught:
            run(src)
        return str(caught.exception)

    def test_a_tool_cut_used_up(self):
        self.assertEqual(self.message("""
            lid = s3d.extrude("lid", s3d.rect(40, 30, r=4), 3)
            box = s3d.box("box", 50, 40, 20)
            s3d.cut(box, s3d.at(lid, z=18.5))
            s3d.at(lid, z=26)
            """), "at: 'lid' was used up by cut(box, ...) on line 3; a tool is consumed by the boolean. "
                  "To keep it, pass s3d.copy(lid) to cut instead.")

    def test_a_tool_joined_then_passed_to_another_verb(self):
        self.assertEqual(self.message("""
            plate = s3d.box("plate", 60, 40, 4)
            boss = s3d.at(s3d.cylinder("boss", 10, 6), z=4)
            s3d.join(plate, boss)
            s3d.fillet(boss, 1)
            """), "fillet: 'boss' was used up by join(plate, ...) on line 3; a tool is consumed by the boolean. "
                  "To keep it, pass s3d.copy(boss) to join instead.")

    def test_a_tool_intersected_and_an_object_merged(self):
        self.assertEqual(self.message("""
            a = s3d.box("a", 10, 10, 10)
            ball = s3d.sphere("ball", 12)
            s3d.intersect(a, ball)
            s3d.at(ball, x=1)
            """), "at: 'ball' was used up by intersect(a, ...) on line 3; a tool is consumed by the boolean. "
                  "To keep it, pass s3d.copy(ball) to intersect instead.")
        s3d.empty_scene()
        self.assertEqual(self.message("""
            a = s3d.box("a", 10, 10, 10)
            b = s3d.at(s3d.box("b", 10, 10, 10), x=20)
            s3d.merge(a, b)
            s3d.group("g", a, b)
            """), "group: 'b' was used up by merge(a, ...) on line 3; an object merged into the first is used up. "
                  "To keep it, pass s3d.copy(b) to merge instead.")

    def test_a_used_up_object_among_others(self):
        self.assertEqual(self.message("""
            lid = s3d.box("lid", 10, 10, 2)
            box = s3d.box("box", 20, 20, 20)
            s3d.cut(box, lid)
            s3d.group("g", *[box, lid])
            """), "group: 'lid' was used up by cut(box, ...) on line 3; a tool is consumed by the boolean. "
                  "To keep it, pass s3d.copy(lid) to cut instead.")

    def test_a_handle_whose_root_is_gone(self):
        self.assertEqual(self.message("""
            lamp = s3d.assembly("lamp", s3d.box("stem", 2, 2, 20))
            s3d.delete(lamp.root)
            s3d.at(lamp, z=1)
            """), "at: 'lamp' was deleted by delete(lamp) on line 2.")

    def test_a_tool_repeated_on_frames_or_cutting_nothing_is_used_up_too(self):
        """However the verb goes — copies on frames, or a tool that only touched — the
        program's tool is gone after it, and says so."""
        self.assertEqual(self.message("""
            plate = s3d.box("plate", 60, 60, 4)
            pin = s3d.cylinder("pin", 4, 10)
            s3d.cut(plate, s3d.at(pin, on=s3d.ring(6, 20)))
            s3d.at(pin, x=1)
            """), "at: 'pin' was used up by at(pin, ...) on line 3; at with a list of frames uses the object up "
                  "and returns one copy per frame: use the list it returns.")
        s3d.empty_scene()
        self.assertIn("'pad' was used up by cut(plate, ...) on line 4", self.message("""
            plate = s3d.box("plate", 60, 60, 4)
            pad = s3d.at(s3d.box("pad", 10, 10, 2), z=4)
            try:
                s3d.cut(plate, pad)          # it only touches: the cut raises, and uses it up all the same
            except ValueError:
                pass
            s3d.fillet(pad, 1)
            """))

    def test_delete_of_a_used_up_object_does_nothing(self):
        """The bpy habit: remove the cutter after the boolean."""
        run("""
            box = s3d.box("box", 20, 20, 20)
            cutter = s3d.at(s3d.cylinder("cutter", 5, 30), z=-5)
            s3d.cut(box, cutter)
            s3d.delete(cutter)
            """)
        self.assertEqual([o.get("s3d_name") for o in bpy.data.objects], ["box"])

    def test_a_deleted_object_passed_to_a_verb(self):
        self.assertEqual(self.message("""
            box = s3d.box("box", 20, 20, 20)
            s3d.delete(box)
            s3d.at(box, x=1)
            """), "at: 'box' was deleted by delete(box) on line 2.")


class PassedTwice(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_one_object_moved_four_times(self):
        """A comprehension written by hand: one object moved four times."""
        with self.assertRaises(ValueError) as caught:
            run("""
                plate = s3d.box("plate", 60, 60, 4)
                rib = s3d.box("rib", 4, 20, 10)
                s3d.join(plate, *[s3d.at(rib, on=f) for f in s3d.ring(4, 20)])
                """)
        self.assertEqual(str(caught.exception),
                         "join: 'rib' is passed 4 times; at moves an object, it does not copy it: "
                         "s3d.at(rib, on=s3d.ring(...)) places one copy per frame")

    def test_a_name_no_variable_could_have(self):
        with self.assertRaises(ValueError) as caught:
            run("""
                plate = s3d.box("Fan plate", 60, 60, 4)
                hole = s3d.cylinder("Radial opening", 5, 10)
                s3d.cut(plate, hole)
                s3d.at(hole, x=1)
                """)
        self.assertEqual(str(caught.exception),
                         "at: 'Radial opening' was used up by cut('Fan plate', ...) on line 3; a tool is consumed by "
                         "the boolean. To keep it, pass s3d.copy(...) to cut instead.")

    def test_the_target_among_the_tools(self):
        plate = s3d.box("plate", 60, 60, 4)
        with self.assertRaisesRegex(ValueError, r"^cut: 'plate' is passed 2 times; "):
            s3d.cut(plate, plate)
        with self.assertRaisesRegex(ValueError, r"^intersect: 'plate' is passed 2 times; "):
            s3d.intersect(plate, plate)
        for verb in (s3d.merge, lambda *o: s3d.group("g", *o), lambda *o: s3d.assembly("a", *o)):
            b = s3d.box("b", 1, 1, 1)
            with self.assertRaisesRegex(ValueError, r"'b' is passed 2 times; "):
                verb(b, b)


class ThrownAway(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def message(self, src):
        with self.assertRaises(ValueError) as caught:
            run(src)
        return str(caught.exception)

    def test_a_placed_profile_nothing_used(self):
        self.assertEqual(self.message("""
            p = s3d.rect(40, 20)
            s3d.at(p, x=100)
            s3d.extrude("a", p, 5)
            """), "extrude: at() on line 2 placed a profile that nothing used; write p = s3d.at(p, ...)")

    def test_a_placed_path_nothing_used(self):
        self.assertEqual(self.message("""
            coil = s3d.helix(20, 5, 30)
            s3d.at(coil, z=10)
            s3d.wire("spring", coil, 2)
            """), "wire: at() on line 2 placed a path that nothing used; write p = s3d.at(p, ...)")

    def test_a_placed_profile_assigned_and_never_read(self):
        self.assertEqual(self.message("""
            p = s3d.rect(40, 20)
            q = s3d.at(p, x=100)
            s3d.export(s3d.group("all", s3d.extrude("a", p, 5)), "unused.glb")
            """), "export: at() on line 2 placed a profile that nothing used; write p = s3d.at(p, ...)")

    def test_placed_profiles_that_are_used(self):
        """Every use counts: a verb, `+` and `-`, a loft's sections, reading the placement."""
        run("""
            plate = s3d.rect(70, 70) - s3d.at(s3d.circle(6), x=-30)
            sections = [s3d.at(s3d.circle(d), z=z) for d, z in ((40, 0), (60, 50))]
            s3d.loft("vase", *sections)
            m = s3d.at(s3d.rect(10, 5), x=3).matrix
            s3d.extrude("plate", plate, 4)
            """)

    def test_the_first_line_is_said_whatever_the_order_in_memory(self):
        self.assertIn("export: at() on line 2 placed", self.message("""
            p = s3d.rect(40, 20)
            a = s3d.at(p, x=1)
            b = s3d.at(p, x=2)
            c = s3d.at(p, x=3)
            s3d.export(s3d.group("all", s3d.extrude("a", p, 5)), "unused.glb")
            """))

    def test_reading_any_part_of_it_is_a_use(self):
        run("""
            for z in (5, 10):
                q = s3d.at(s3d.helix(20, 5, 30), z=z)
                n = 1 if q.closed else 0
            s3d.box("b", 1, 1, 1)
            """)

    def test_an_empty_scene_starts_clean(self):
        run("""
            s3d.at(s3d.rect(40, 20), x=100)
            """)
        s3d.empty_scene()
        s3d.box("b", 1, 1, 1)
        held = s3d.at(s3d.rect(40, 20), x=100)
        s3d.empty_scene()
        del held                         # the last program's value, freed in the new scene
        s3d.box("b", 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
