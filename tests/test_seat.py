"""`seat`: the cavity a part leaves in its host, at rest; the part untouched."""
import math
import unittest

import bpy  # noqa: F401 — before bmesh
import bmesh
from mathutils import Matrix

import sand3d as s3d
from sand3d import _core
from test_regressions import manifold


def volume(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    v = bm.calc_volume()
    bm.free()
    return v


def mesh(ob):
    return [tuple(v.co) for v in ob.data.vertices], [tuple(p.vertices) for p in ob.data.polygons]


class Seat(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_the_host_loses_the_volume_it_shares_and_the_part_is_untouched(self):
        host = s3d.box("console", 100, 100, 100)
        knob = s3d.at(s3d.cylinder("knob", 40, 30), z=80)
        before, placed = mesh(knob), Matrix(knob.matrix_basis)
        got = s3d.seat(host, knob)
        self.assertIs(got, host)
        self.assertAlmostEqual(volume(host), 100 ** 3 - math.pi * 20 ** 2 * 20, delta=0.5 * 100 ** 3 * 0.001)
        self.assertEqual(mesh(knob), before)
        self.assertEqual(Matrix(knob.matrix_basis), placed)
        self.assertTrue(manifold(host))
        self.assertIn("s3d_face:cut", host.data.attributes)
        self.assertIn("s3d_edge:knob", host.data.attributes)       # the edges of the seat carry the part's name

    def test_a_group_seats_every_solid_in_it(self):
        host = s3d.box("plate", 100, 100, 10)
        pins = s3d.group("pins", *s3d.at(s3d.cylinder("pin", 10, 30), on=s3d.grid(2, 1, 50), z=-10))
        s3d.seat(host, pins)
        n = 24                                                       # a 10 mm circle's segments
        self.assertAlmostEqual(volume(host), 100 * 100 * 10 - 2 * n / 2 * 5 ** 2 * math.sin(2 * math.pi / n) * 10,
                               places=2)
        self.assertEqual(len(pins.children), 2)

    def test_the_cavity_is_cut_at_rest_whatever_the_pose(self):
        cab = s3d.box("cab", 600, 600, 850)
        drawer = s3d.at(s3d.box("drawer", 100, 150, 60), x=260, z=700)
        s3d.joint("slide", drawer, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
        s3d.pose(drawer, 150)
        s3d.seat(cab, drawer)
        xs = [v.co.x for v in cab.data.vertices if 699 < v.co.z < 761]
        self.assertAlmostEqual(min(xs), 210, places=4)              # where the drawer rests, not where it is
        from sand3d.joints import joints_of
        self.assertEqual(joints_of(drawer)[0]["value"], 150.0)

    def test_a_moving_host_takes_its_seat_at_rest(self):
        cab = s3d.box("cab", 600, 600, 850)
        door = s3d.at(s3d.box("door", 20, 560, 800), x=310, z=25)
        knob = s3d.at(s3d.cylinder("knob", 40, 30), on=door.face("+X").move(x=200, y=400).offset(-10))
        s3d.joint("hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
        s3d.joint("dial", knob, door.face("+X").move(x=200, y=400), kind="continuous")
        s3d.pose(door, 90)
        s3d.pose(knob, 30)
        s3d.seat(door, knob)
        cavity = [v.co for v in door.data.vertices if abs(v.co.y - 200) < 21 and abs(v.co.z - 400) < 21]
        self.assertTrue(cavity)
        self.assertAlmostEqual(min(v.x for v in cavity), 0, places=4)      # in the door's own space, at rest

    def test_a_lens_seated_in_a_rim_is_clean(self):
        """Glasses: one outline for the rim and the lens, then the lens seated."""
        R, C = 83.0, (15.3, 76.8, 31.9)
        eye = [(-28, 40), (-18, 46), (-4, 46), (18, 44), (27, 38), (24, 22), (18, 10), (4, 1.5), (-16, 3),
               (-26, 10), (-28, 25)]
        on_blank = lambda x, z: (x, C[1] - math.sqrt((R - 1) ** 2 - (x - C[0]) ** 2 - (z - C[2]) ** 2), z)  # noqa
        rim = s3d.wire("rim", s3d.spline([on_blank(x, z) for x, z in eye]), d=3.5, closed=True)
        brow = s3d.wire("brow", s3d.spline([(-32, 14, 40), (-18, 1, 46), (-4, -4, 46), (12, -5, 45), (29, -6, 38),
                                            (35, -7, 38)]), d=5)
        s3d.join(rim, brow)
        blank = s3d.at(s3d.shell(s3d.sphere("lens", 2 * R), 2), x=C[0], y=C[1], z=C[2] - R)
        lens = s3d.intersect(blank, s3d.extrude("edge", s3d.spline(eye, closed=True), R + 10,
                                                on=s3d.Frame((0, C[1], 0), normal=(0, -1, 0))))
        before = mesh(lens)
        s3d.seat(rim, lens)
        self.assertTrue(manifold(rim))
        keys = [tuple(sorted(p.vertices)) for p in rim.data.polygons]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(mesh(lens), before)


class Errors(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.host = s3d.box("host", 100, 100, 100)

    def test_nothing_in_contact(self):
        for z, what in ((500, "apart"), (100, "touching")):
            with self.subTest(what=what):
                part = s3d.at(s3d.box("part", 10, 10, 10), z=z)
                with self.assertRaises(ValueError) as caught:
                    s3d.seat(self.host, part)
                self.assertEqual(str(caught.exception), "seat: 'part' shares no volume with 'host' at rest — they "
                                                        "miss each other, or only touch: there is nothing to seat")
                self.assertEqual(len([o for o in bpy.data.objects if o.get("s3d_name") == "part"]), 1)
                s3d.delete(part)

    def test_the_host_and_the_part(self):
        with self.assertRaisesRegex(ValueError, r"^seat: 'host' is both the host and the part$"):
            s3d.seat(self.host, self.host)
        group = s3d.group("g", self.host, s3d.box("b", 1, 1, 1))
        with self.assertRaisesRegex(ValueError, r"^seat: 'host' is inside the part 'g'$"):
            s3d.seat(self.host, group)
        with self.assertRaisesRegex(TypeError, r"^seat: 'g' is a group, not a solid"):
            s3d.seat(group, self.host)
        with self.assertRaisesRegex(TypeError, r"^seat: the part is a solid, a group or an assembly, not Frame$"):
            s3d.seat(self.host, s3d.Frame())
        with self.assertRaisesRegex(ValueError, r"^seat: 'void' holds no solid$"):
            s3d.seat(self.host, s3d.group("void"))


if __name__ == "__main__":
    unittest.main()
