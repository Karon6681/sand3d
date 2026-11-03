"""Each verb checks its arguments and says what to write: one test per misuse."""
import math
import unittest

import sand3d as s3d


class Arguments(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def says(self, error, message, call, *args, **kwargs):
        with self.assertRaises(error) as caught:
            call(*args, **kwargs)
        self.assertEqual(str(caught.exception), message)

    def test_a_counterbore_is_a_pair(self):
        """Seen in written programs: `hole(..., cbore=4.5)` ×4 — 'float' object is not subscriptable."""
        plate = s3d.box("plate", 40, 40, 10)
        self.says(TypeError, "hole: cbore is (diameter, depth), not 4.5", s3d.hole, plate, 3, on=(0, 0), cbore=4.5)
        self.says(TypeError, "hole: csk is a diameter, or (diameter, angle), not (6,)", s3d.hole, plate, 3,
                  on=(0, 0), csk=(6,))

    def test_a_frame_is_not_a_point(self):
        """Seen in written programs: `for x, y in s3d.grid(...)` ×3 — cannot unpack non-iterable Frame object."""
        with self.assertRaises(TypeError) as caught:
            for x, y in s3d.grid(2, 1, 10):
                pass
        self.assertEqual(str(caught.exception), "a Frame is not a point: use f.origin (f.origin.x, f.origin.y, "
                                                "f.origin.z)")

    def test_a_spline_has_two_end_tangents(self):
        """Seen in written programs: `spline(pts, tangents=[t0, t1, …, tn])` ×2 — too many values to unpack."""
        pts = [(0, 0), (10, 5), (20, 0), (30, 5)]
        self.says(ValueError, "spline: tangents is (start, end), the directions at its two ends, not 4 directions",
                  s3d.spline, pts, tangents=[(1, 0)] * 4)

    def test_an_arc_is_given_one_way(self):
        """Seen in written programs: `arc(center=, start=, end=, through=)` ×2 — 'NoneType' object is not iterable."""
        self.says(ValueError, "arc: either center, r, start and end, or through and to (inside poly), not both",
                  s3d.arc, center=(0, 0), r=5, start=270, end=90, through=(0, 5))
        self.says(ValueError, "arc: through and to go together, inside poly: arc(through=(x, y), to=(x, y))",
                  s3d.arc, through=(0, 5))

    def test_loft_sections_are_placed_profiles(self):
        """Seen in written programs: `loft(name, (profile, Frame), …)` or solids as sections ×2."""
        self.says(TypeError, "loft: a section is a profile placed with at — s3d.at(profile, on=frame) — not a "
                             "(profile, Frame) pair", s3d.loft, "l", (s3d.circle(10), s3d.Frame()),
                  (s3d.circle(10), s3d.Frame((0, 0, 5))))
        box = s3d.box("lid", 10, 10, 2)
        self.says(TypeError, "loft: a section is a profile placed with at, not a solid ('lid')", s3d.loft, "l",
                  s3d.circle(10), box)

    def test_an_origin_has_three_coordinates(self):
        """Seen in written programs: `Frame((x, y))` — Vector addition: vectors must have the same dimensions."""
        self.says(ValueError, "Frame: an origin is (x, y, z), not (0, 30)", s3d.Frame, (0, 30))
        self.says(ValueError, "Frame: a normal is (x, y, z), not (0, 1)", s3d.Frame, (0, 0, 0), (0, 1))

    def test_a_solid_moves_with_transform(self):
        """Seen in written programs: `s3d.at(...).move(...)` on an object ×2 — 'Object' object has no attribute 'move'."""
        box = s3d.box("b", 1, 1, 1)
        for method in ("move", "turn", "offset"):
            with self.subTest(method):
                with self.assertRaises(TypeError) as caught:
                    getattr(box, method)(1)
                self.assertEqual(str(caught.exception),
                                 f"'b' is a solid: it moves with s3d.transform(ob, x=…, rz=…), or s3d.at(ob, …) "
                                 f"places it; .{method}() is a frame's")

    def test_edge_names_say_which_maker_names_what(self):
        """Seen in written programs: `chamfer(..., edges="end")` on an extrusion ×9 (`end` ×5, `vertical` ×2…)."""
        lip = s3d.extrude("lip", s3d.rect(20, 10), 4)
        self.says(ValueError, "lip has no edge named ['end']; its edge names: ['bottom', 'lip.bottom', 'lip.side', "
                              "'lip.top', 'side', 'top']. extrude names its caps top and bottom; start and end are "
                              "a sweep's, a loft's or a partial revolve's", s3d.chamfer, lip, 1, edges="end")
        self.says(ValueError, "lip has no edge named ['vertical']; its edge names: ['bottom', 'lip.bottom', "
                              "'lip.side', 'lip.top', 'side', 'top']. Edges go by the names their maker gave them, "
                              "not by direction: an extrude's walls meet at side", s3d.fillet, lip, 1, edges="vertical")
        self.says(ValueError, "lip has no edge named ['sides']; its edge names: ['bottom', 'lip.bottom', "
                              "'lip.side', 'lip.top', 'side', 'top']. Did you mean 'side'?", s3d.fillet, lip, 1,
                  edges="sides")
        tab = s3d.extrude("tab", s3d.rect(20, 10, r=3), 4)
        self.says(ValueError, "tab has no edge named ['side']; its edge names: ['bottom', 'tab.bottom', 'tab.top', "
                              "'top']. Its outline turns smoothly all round (a rounded rectangle, a circle): no corner "
                              "makes a side edge", s3d.fillet, tab, 1, edges="side")
        bar = s3d.sweep("bar", s3d.rect(4, 4), [(0, 0, 0), (0, 0, 30)])
        self.says(ValueError, "bar has no edge named ['top']; its edge names: ['bar.end', 'bar.seam', 'bar.start', "
                              "'end', 'seam', 'start']. A sweep, a loft or a partial revolve names its caps start and "
                              "end; top and bottom are an extrude's", s3d.chamfer, bar, 1, edges="top")
        self.says(ValueError, "lip has no face named ['end']; its face names: ['bottom', 'lip.bottom', 'lip.side', "
                              "'lip.top', 'side', 'top']. extrude names its caps top and bottom; start and end are "
                              "a sweep's, a loft's or a partial revolve's", s3d.shell, lip, 1, open="end")

    def test_a_chamfer_angle_is_in_degrees(self):
        """Seen ×5, silent: bpy's angle_limit, in radians, passed as the chamfer's face angle."""
        plate = s3d.box("plate", 40, 40, 10)
        self.says(ValueError, "chamfer: angle is the face angle in degrees, from 1 up to 90, not 0.01745 — radians? "
                              "give degrees", s3d.chamfer, plate, 2, angle=math.radians(1))
        self.says(ValueError, "chamfer: angle is the face angle in degrees, from 1 up to 90, not 90", s3d.chamfer,
                  plate, 2, angle=90)
        s3d.chamfer(plate, 1, edges="top", angle=30)

    def test_extrude_is_drawn_on_one_frame(self):
        message = "extrude: on is one frame; for one solid per frame, place copies: s3d.at(s3d.extrude(...), on=[frames])"
        self.says(TypeError, message, s3d.extrude, "boss", s3d.circle(4), 3, on=s3d.grid(2, 1, 10))
        self.says(TypeError, message, s3d.extrude, "boss", s3d.circle(4), until=s3d.Frame((0, 0, 5)),
                  on=s3d.grid(2, 1, 10))

    def test_a_verb_on_solids_takes_solids(self):
        plate = s3d.box("plate", 40, 40, 4)
        lamp = s3d.assembly("lamp", s3d.box("stem", 2, 2, 20))
        group = "'lamp' is a group, not a solid: give its parts, or join them into one first"
        self.says(TypeError, "join: " + group, s3d.join, plate, lamp.root)
        self.says(TypeError, "cut: " + group, s3d.cut, plate, [lamp.root])
        self.says(TypeError, "hole: " + group, s3d.hole, lamp.root, 2)
        self.says(TypeError, "pocket: " + group, s3d.pocket, lamp.root, s3d.circle(2), depth=1)
        self.says(TypeError, "fillet: " + group, s3d.fillet, lamp.root, 1)
        self.assertEqual(sorted(o.name for o in __import__("bpy").data.objects), ["lamp", "plate", "stem"],
                         "nothing made, nothing used up")
        self.says(TypeError, "cut: a profile is not a solid: extrude it first, s3d.extrude(name, profile, h)",
                  s3d.cut, plate, s3d.circle(3))

    def test_what_used_to_work_still_does(self):
        from mathutils import Vector
        plate = s3d.box("plate", 40, 40, 10)
        s3d.hole(plate, 3, on=(0, 0), cbore=Vector((6, 2)))
        s3d.hole(plate, 3, on=(10, 0), cbore=())
        s3d.extrude("e", s3d.circle(2), 3, on=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        from sand3d.profile import arc
        arc(through=(1, 1), to=(2, 0))      # from inside the library: no program call to judge

    def test_a_pocket_takes_one_profile(self):
        plate = s3d.box("plate", 40, 40, 4)
        self.says(TypeError, "pocket: one profile per pocket; to repeat it, give on=[frames]", s3d.pocket, plate,
                  [s3d.circle(4), s3d.at(s3d.circle(4), x=10)], depth=1)


if __name__ == "__main__":
    unittest.main()
