"""Edge treatments and shell: exact where geometry gives a closed form, closed always."""
import math
import unittest

import bpy  # noqa: F401 — before bmesh, which bpy provides
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import sand3d as s3d
from sand3d.edit import _offset
from test_solids import closed_volume, edges, faces


def walls(ob):
    """The wall at every face: a ray from its centre inwards to the opposite sheet."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    out = []
    for f in bm.faces:
        c, n = f.calc_center_median(), f.normal
        hit = tree.ray_cast(c - n * 0.05, -n)   # a curved quad is not flat: start clear of its own triangles
        if hit[0] is not None:
            out.append((c - hit[0]).length)
    bad = sum(len(e.link_faces) != 2 for e in bm.edges)
    bm.free()
    return out, bad


class EdgeTreatments(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def assertSolid(self, ob, volume, rel=0.005):
        closed, v = closed_volume(ob)
        self.assertTrue(closed, f"{ob.name} is not closed")
        self.assertAlmostEqual(v, volume, delta=abs(volume) * rel, msg=ob.name)

    def test_fillet_by_name(self):
        ob = s3d.fillet(s3d.box("b", 40, 30, 20), 3, edges="top")
        self.assertSolid(ob, 24000 - (1 - math.pi / 4) * 9 * 140)
        self.assertGreater(faces(ob, "fillet"), 0)
        self.assertEqual(edges(ob, "top"), 0, "a rounded edge is no edge")
        self.assertEqual(edges(ob, "bottom"), 4)

    def test_fillet_closes_round_a_loft_of_superellipses(self):
        """A station loft's cap edge bends by 98° here and 109° there; bevelled a bend at a
        time, its pieces overlapped and left it open, or closed by slivers the GLB folds open."""
        for n in (2, 4):
            for r, which in ((50, "start"), (50, "end"), (10, "start")):
                with self.subTest(n=n, r=r, edges=which):
                    s3d.empty_scene()
                    tub = s3d.loft("tub", s3d.station(0, 510, 1110, n=n), s3d.station(602, 686, 1520, n=n))
                    s3d.fillet(tub, r, edges=which)
                    bm = bmesh.new()
                    bm.from_mesh(tub.data)
                    self.assertEqual(sum(len(e.link_faces) != 2 for e in bm.edges), 0, "open")
                    self.assertGreater(min(e.calc_length() for e in bm.edges), 1e-3, "a sliver")
                    bm.free()
                    self.assertIn("s3d_face:fillet", tub.data.attributes)
                    closed, v = closed_volume(tub)
                    self.assertTrue(closed)

    def test_a_fillet_redone_keeps_each_edge_its_radius(self):
        """A computer casing comes back open from
        a fillet bevelled a bend at a time. The pass that closed it set every edge of the network
        back by their mean: the square edges at 0.81 r, shallow ones up to 15 r. Redone, each edge
        keeps its own setback, so the fillet's facets turn at radius r."""
        outline = s3d.poly([(-217, 0), (209, 0), (212, 18), (212, 172), (208, 175), (-203, 189.2), (-204, 184.6),
                            (-195, 182.6), (-217, 84.6)])
        body = s3d.extrude("housing", outline, h=444, on=s3d.Frame(origin=(-222, 0, 0), normal=(1, 0, 0), up=(0, 0, 1)))
        cross = s3d.poly([(-196, 0), (193, 0), (196, 16), (218, 28), (222, 70), (218, 173), (209, 209.2),
                          (-209, 209.2), (-218, 177), (-222, 70), (-219, 28), (-196, 15)])
        tool = s3d.extrude("side_taper", cross, h=852, both=True, on=s3d.Frame(normal=(0, -1, 0), up=(0, 0, 1)))
        casing = s3d.fillet(s3d.intersect(body, tool), 2)
        bm = bmesh.new()
        bm.from_mesh(casing.data)
        bm.normal_update()
        self.assertEqual(sum(len(e.link_faces) != 2 for e in bm.edges), 0, "open")
        self.assertGreater(min(e.calc_length() for e in bm.edges), 1e-3, "a sliver")
        # the radius a fillet facet turns at: its width across, over the angle to the next facet
        rounded = bm.faces.layers.bool["s3d_face:fillet"]
        radii = []
        for e in bm.edges:
            f, g = e.link_faces
            if not (f[rounded] and g[rounded]) or len(f.verts) != 4 or f.normal.angle(g.normal, 0) < math.radians(2):
                continue
            across = next((o for o in f.edges if not set(o.verts) & set(e.verts)), None)
            if across is not None:
                width = ((e.verts[0].co + e.verts[1].co) / 2 - (across.verts[0].co + across.verts[1].co) / 2).length
                radii.append(width / (2 * math.sin(f.normal.angle(g.normal, 0) / 2)))
        bm.free()
        radii.sort()
        self.assertGreater(len(radii), 1000)
        self.assertAlmostEqual(radii[len(radii) // 2], 2, delta=0.05, msg="the median facet turns at r")

    def test_chamfer_equal_two_distances_and_angle(self):
        corner = 4 * 8 / 3
        self.assertSolid(s3d.chamfer(s3d.box("a", 40, 30, 20), 2, edges="top"), 24000 - 280 + corner, 1e-9)
        self.assertSolid(s3d.chamfer(s3d.box("b", 40, 30, 20), 2, edges="top", d2=4),
                         24000 - 560 + 4 * 16 / 3, 1e-9)
        d2 = 2 * math.tan(math.radians(60))
        self.assertSolid(s3d.chamfer(s3d.box("c", 40, 30, 20), 2, edges="top", angle=60),
                         24000 - 140 * d2 + 4 * 4 * d2 / 3, 1e-6)

    def test_chamfer_names_its_edges(self):
        ob = s3d.chamfer(s3d.box("b", 40, 30, 20), 2, edges="top")
        self.assertEqual(edges(ob, "top"), 4, "the top face is still bounded by top edges")
        self.assertEqual(edges(ob, "chamfer"), 12)

    def test_hole_mouths_take_a_chamfer(self):
        plate = s3d.box("p", 60, 40, 8)
        s3d.hole(plate, 10, on=(0, 0))
        s3d.chamfer(plate, 1, edges="hole")
        self.assertTrue(closed_volume(plate)[0])
        self.assertGreater(faces(plate, "chamfer"), 0)

    def test_unknown_name_says_what_exists(self):
        with self.assertRaises(ValueError) as caught:
            s3d.fillet(s3d.box("x", 10, 10, 10), 1, edges="rim")
        self.assertIn("top", str(caught.exception))

    def test_soften(self):
        ob = s3d.soften(s3d.box("s", 40, 30, 20), 2)
        self.assertSolid(ob, 23689.5, rel=0.002)

    def test_shell(self):
        self.assertSolid(s3d.shell(s3d.box("o", 40, 30, 20), 2, open="top"), 24000 - 36 * 26 * 18, 1e-6)
        self.assertSolid(s3d.shell(s3d.box("c", 40, 30, 20), 2), 24000 - 36 * 26 * 16, 1e-6)
        cup = s3d.shell(s3d.cylinder("cup", 40, 50), 2, open="top")
        self.assertSolid(cup, math.pi * (400 - 324) * 50 + math.pi * 324 * 2, rel=0.01)
        self.assertGreater(edges(cup, "shell"), 0, "the rim of the opening is named")
        rounded = s3d.shell(s3d.fillet(s3d.box("r", 40, 30, 20), 4, edges="bottom side"), 1.5, open="top")
        self.assertTrue(closed_volume(rounded)[0])

    def test_shell_opens_the_face_looking_a_way(self):
        """Open= reads a direction as ob.face does — the flat face furthest that way."""
        def mesh(ob):
            return [tuple(v.co) for v in ob.data.vertices], [tuple(q.vertices) for q in ob.data.polygons]

        crate = s3d.shell(s3d.box("crate", 60, 40, 30), 3, open="+Y")
        self.assertSolid(crate, 60 * 40 * 30 - 54 * 37 * 24, 1e-6)       # five sides, open toward +Y
        rim = sum(q.area for q in crate.data.polygons if q.normal.y > 0.999 and abs(q.center.y - 20) < 1e-6)
        self.assertAlmostEqual(rim, 60 * 30 - 54 * 24, delta=1e-3)
        self.assertSolid(s3d.shell(s3d.box("sleeve", 60, 40, 30), 3, open="-Y +Y"), 60 * 40 * 30 - 54 * 40 * 24, 1e-6)
        # the names keep their meaning, and say the same faces as the directions
        makers = [(lambda: s3d.box("b", 60, 40, 30), "top", "+Z"),
                  (lambda: s3d.extrude("e", s3d.rect(50, 30, r=6), 20), "bottom", "-Z"),
                  (lambda: s3d.loft("l", s3d.station(0, 40, 30, n=4), s3d.station(50, 30, 20, n=4)), "end", "+Z")]
        for make, name, way in makers:
            with self.subTest(name=name):
                s3d.empty_scene()
                by_name = mesh(s3d.shell(make(), 2, open=name))
                s3d.empty_scene()
                self.assertEqual(mesh(s3d.shell(make(), 2, open=way)), by_name)
        # a loft along Y, its sections stood up: open on top, and mixed with a name
        s3d.empty_scene()
        hull = s3d.loft("hull", s3d.at(s3d.rect(40, 20), rx=90), s3d.at(s3d.rect(40, 20), y=100, rx=90))
        self.assertSolid(s3d.shell(hull, 2, open="+Z start"), 40 * 20 * 100 - 36 * 18 * 98, 1e-6)
        # a curved side has no flat face: refused, saying how to open it
        s3d.empty_scene()
        tube = s3d.loft("tube", s3d.at(s3d.ellipse(40, 20), rx=90), s3d.at(s3d.ellipse(40, 20), y=100, rx=90))
        with self.assertRaisesRegex(ValueError, r"shell: 'tube' has no flat face looking \+Z .*open=\"cut\""):
            s3d.shell(tube, 2, open="+Z")

    def test_shell_refuses_a_curved_side_made_of_facets(self):
        """A superellipse is 96 facets, and the one or two at its top look +Z within 1°.
        They are not a face: opened, they made a slot 25 mm wide along a 400 mm hull, without a word."""
        for n, w in ((4, 400), (3, 400), (2, 1000)):
            with self.subTest(n=n, w=w):
                s3d.empty_scene()
                hull = s3d.loft("hull", s3d.at(s3d.superellipse(w, 300, n=n), rx=90),
                                s3d.at(s3d.superellipse(w, 300, n=n), y=1000, rx=90))
                with self.assertRaisesRegex(ValueError, r"shell: 'hull' has no flat face looking \+Z: the faces "
                                                        r"furthest that way are facets of a curved side.*open=\"cut\""):
                    s3d.shell(hull, 10, open="+Z")
        # a flat face that runs on smoothly into a fillet, or into a rounded corner, is still a face
        def mesh(ob):
            return [tuple(v.co) for v in ob.data.vertices], [tuple(q.vertices) for q in ob.data.polygons]
        s3d.empty_scene()
        by_name = mesh(s3d.shell(s3d.fillet(s3d.box("b", 200, 100, 60), 20, edges="top"), 3, open="top"))
        s3d.empty_scene()
        self.assertEqual(mesh(s3d.shell(s3d.fillet(s3d.box("b", 200, 100, 60), 20, edges="top"), 3, open="+Z")),
                         by_name)
        s3d.empty_scene()
        deck = s3d.loft("deck", s3d.at(s3d.rect(400, 300, r=40), rx=90), s3d.at(s3d.rect(400, 300, r=40), y=1000, rx=90))
        self.assertTrue(closed_volume(s3d.shell(deck, 10, open="+Z"))[0])

    def test_shell_offset_is_t_where_the_planes_are_nearly_parallel(self):
        """Four normals 3.6° apart, as at any vertex of a smooth loft, make a system of
        condition ~1e6 — float32 `Matrix.inverted()` answered 44 t on this one."""
        normals = [(-0.8079, -0.5711, 0.1454), (-0.7698, -0.6219, 0.1434),
                   (-0.8097, -0.5704, 0.1380), (-0.7717, -0.6213, 0.1360)]
        planes = [(Vector(n).normalized(), -1.0) for n in normals]
        for t in (0.5, 2.0, 8.0):
            x = _offset(planes, t)
            self.assertAlmostEqual(x.length / t, 1.0, delta=0.01)
            for n, s in planes:
                self.assertAlmostEqual(-n.dot(x) / t, 1.0, delta=0.01, msg="t inside every face plane")

    def test_shell_offset_corners_stay_exact(self):
        """The planes that do pin a direction keep their exact answer: a cube corner moves
        t along each axis, two square planes t along each of theirs and nothing along the edge."""
        axes = [(Vector(a), -1.0) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        self.assertAlmostEqual((_offset(axes, 3.0) + Vector((3, 3, 3))).length, 0, places=12)
        self.assertAlmostEqual((_offset(axes[:2], 3.0) + Vector((3, 3, 0))).length, 0, places=12)

    def test_shell_wall_of_a_smooth_loft_is_the_thickness_asked(self):
        """Five superellipse stations, no corner on the side: the wall ranged over two
        orders of magnitude around `t`, and pinched shut where it fell."""
        def station(z, w, d):
            return s3d.at(s3d.superellipse(w, d, n=2), z=z)
        for t in (0.5, 2.0):
            s3d.empty_scene()
            jug = s3d.loft("jug", station(7, 144, 146), station(12, 147, 148), station(20, 146, 147),
                           station(110, 125, 128), station(121, 123, 125))
            thick, bad = walls(s3d.shell(jug, t))
            self.assertEqual(bad, 0, "every edge of the shell is shared by two faces")
            self.assertGreater(min(thick), 0.95 * t, f"t={t}: the thinnest wall")
            self.assertAlmostEqual(sorted(thick)[len(thick) // 2], t, delta=0.01 * t)


if __name__ == "__main__":
    unittest.main()
