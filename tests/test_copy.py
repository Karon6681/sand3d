"""copy: an independent copy, deep, with the same mesh, names and placement."""
import unittest

import bpy

import sand3d as s3d
from harness import world
from test_solids import closed_volume, edges


def mesh_of(ob):
    me = ob.data
    names = sorted((a.name, tuple(d.value for d in a.data)) for a in me.attributes if a.name.startswith("s3d_"))
    return [tuple(v.co) for v in me.vertices], [tuple(p.vertices) for p in me.polygons], names


class Copy(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_same_mesh_names_and_placement(self):
        plate = s3d.at(s3d.extrude("plate", s3d.rect(40, 20, r=3), 5), x=10, z=4, rz=30)
        s3d.hole(plate, 4, on=(5, 0))
        twin = s3d.copy(plate)
        self.assertIsNot(twin, plate)
        self.assertEqual(mesh_of(twin), mesh_of(plate))
        self.assertEqual(twin["s3d_name"], "plate")
        self.assertEqual(world(twin), world(plate))
        self.assertIsNot(twin.data, plate.data)

    def test_the_original_is_untouched_and_independent(self):
        lid = s3d.extrude("lid", s3d.rect(40, 30, r=4), 3)
        before = mesh_of(lid)
        twin = s3d.copy(lid)
        s3d.hole(twin, 5, on=(0, 0))
        s3d.at(twin, z=16)
        self.assertEqual(mesh_of(lid), before)
        self.assertEqual(tuple(lid.matrix_basis.translation), (0.0, 0.0, 0.0))

    def test_a_tool_kept_with_copy(self):
        """The report's lid: a copy cuts the seat, the lid itself stays — the same meshes as
        building the lid twice."""
        def build(with_copy):
            s3d.empty_scene()
            box = s3d.box("box", 50, 40, 20)
            if with_copy:
                lid = s3d.extrude("lid", s3d.rect(40, 30, r=4), 3)
                s3d.cut(box, s3d.at(s3d.copy(lid), z=18.5))
                s3d.at(lid, z=26)
            else:
                s3d.cut(box, s3d.at(s3d.extrude("lid", s3d.rect(40, 30, r=4), 3), z=18.5))
                lid = s3d.at(s3d.extrude("lid", s3d.rect(40, 30, r=4), 3), z=26)
            return mesh_of(box), mesh_of(lid), world(lid)
        self.assertEqual(build(True), build(False))

    def test_copies_keep_the_name_edges_answer_to(self):
        """Same sand3d name: after a join, `fillet(edges="leg")` reaches every copy."""
        top = s3d.box("top", 60, 30, 4)
        leg = s3d.at(s3d.cylinder("leg", 6, 40), x=-20, z=-40)
        s3d.join(top, leg, s3d.at(s3d.copy(leg), x=20, z=-40))
        self.assertEqual(edges(top, "leg"), 2 * edges(s3d.cylinder("probe", 6, 40), "bottom"))

    def test_a_group_is_copied_with_its_children(self):
        a = s3d.at(s3d.box("a", 10, 10, 10), x=-10)
        b = s3d.at(s3d.sphere("b", 8), x=10, z=5)
        g = s3d.at(s3d.group("pair", a, b), y=30, rz=45)
        twin = s3d.copy(g)
        self.assertEqual(twin.type, "EMPTY")
        self.assertEqual(twin["s3d_name"], "pair")
        self.assertEqual(world(twin), world(g))
        self.assertEqual(len(twin.children), 2)
        self.assertEqual(len(g.children), 2, "the original keeps its children")
        pairs = sorted(((c["s3d_name"], tuple(map(tuple, world(c)))) for c in twin.children))
        originals = sorted(((c["s3d_name"], tuple(map(tuple, world(c)))) for c in g.children))
        self.assertEqual(pairs, originals)
        self.assertFalse({c.data for c in twin.children} & {c.data for c in g.children})
        s3d.at(twin, x=100)
        moved = sorted(((c["s3d_name"], tuple(map(tuple, world(c)))) for c in g.children))
        self.assertEqual(moved, originals, "moving the copy leaves the original where it was")

    def test_a_handle_is_copied_with_its_anchors(self):
        table = s3d.assembly("table", s3d.at(s3d.box("top", 100, 60, 4), z=68),
                             s3d.cylinder("leg", 5, 68), top=72, seat=s3d.Frame((0, 0, 72)))
        s3d.attach(table, "lamp", s3d.at(s3d.dome("shade", 20, 10), z=table.top))
        twin = s3d.copy(table)
        self.assertIsInstance(twin, s3d.Handle)
        self.assertIsNot(twin.root, table.root)
        self.assertEqual(twin.top, 72)
        self.assertEqual(tuple(twin.seat.origin), (0.0, 0.0, 72.0))
        self.assertIsNot(twin.seat, table.seat)
        self.assertEqual(len(twin.root.children), 3)   # top, leg and the attached lamp
        self.assertEqual(len([o for o in bpy.data.objects if o.get("s3d_name") == "shade"]), 2)

    def test_anchors_are_copied_exactly_and_objects_map_to_their_twins(self):
        """A frame anchor is the same frame to the bit; an anchor that is a part of the
        assembly is that part's copy; namedtuples and dicts are copied too."""
        from collections import namedtuple
        Corner = namedtuple("Corner", "x y")
        lid = s3d.at(s3d.box("lid", 20, 20, 2), z=28)
        seat = s3d.Frame((1, 2, 3)).turn(rx=17.3, ry=-41.9, rz=73.1)
        h = s3d.assembly("crate", s3d.box("body", 20, 20, 28), lid, seat=seat, lid=lid, corner=Corner(1, 2),
                         holes={"a": s3d.Frame((0, 0, 5))}, sizes=[1, 2.5])
        twin = s3d.copy(h)
        self.assertEqual(tuple(map(tuple, twin.seat.matrix)), tuple(map(tuple, seat.matrix)))
        self.assertIsNot(twin.seat.origin, seat.origin)
        self.assertIsNot(twin.lid, lid)
        self.assertIs(twin.lid.parent, twin.root)
        self.assertEqual((twin.corner, type(twin.corner)), (Corner(1, 2), Corner))
        self.assertIsNot(twin.holes["a"], h.holes["a"])
        self.assertEqual(twin.sizes, [1, 2.5])

    def test_a_handle_subclass_stays_one(self):
        class Crate(s3d.Handle):
            def volume(self):
                return 42
        h = Crate(s3d.group("c", s3d.box("b", 1, 1, 1)), top=1)
        self.assertEqual(s3d.copy(h).volume(), 42)

    def test_a_copy_of_a_merged_part_is_merged(self):
        """The copy is the same object: what `merge` marked stays marked."""
        m = s3d.merge(s3d.box("m", 10, 10, 10), s3d.at(s3d.box("n", 10, 10, 10), x=5))
        twin = s3d.copy(m)
        self.assertTrue(twin.get("s3d~merged"))
        s3d.hole(twin, 3, on=(0, 0))
        self.assertTrue(closed_volume(twin)[0])
        self.assertAlmostEqual(closed_volume(twin)[1], 15 * 10 * 10 - 3.14 * 1.5 ** 2 * 10, delta=2)


if __name__ == "__main__":
    unittest.main()
