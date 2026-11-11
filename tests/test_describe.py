"""describe: what a part holds, read back as a model would before its next verb."""
import unittest

import bpy
import bmesh

import sand3d as s3d


class Describe(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_box(self):
        d = s3d.describe(s3d.box("box", 100, 60, 10))
        self.assertEqual(d, {
            "name": "box", "bbox": [[-50.0, -30.0, 0.0], [50.0, 30.0, 10.0]], "size": [100.0, 60.0, 10.0],
            "volume": 60000.0, "open_edges": 0, "nonmanifold_edges": 0,
            "faces": {n: {"area": a, "frame": True} for n, a in (("bottom", 6000.0), ("side", 3200.0), ("top", 6000.0))},
            "edges": {n: {"length": l} for n, l in (("bottom", 320.0), ("side", 40.0), ("top", 320.0))},
            "joints": [], "material": None, "anchors": [], "parts": []})

    def test_names_come_and_go_with_the_verbs(self):
        """A hole adds its names, curved ones giving no frame; a fillet rounds away
        most of the edges it was given and names its faces."""
        plate = s3d.box("plate", 80, 40, 6)
        before = s3d.describe(plate)["edges"]["top"]["length"]
        s3d.hole(plate, 6, on=plate.face("top"))
        s3d.fillet(plate, 1, edges="top")
        d = s3d.describe(plate)
        self.assertEqual(d["faces"]["hole.side"]["frame"], False)
        self.assertTrue(3.1 * 6 * 4 < d["faces"]["hole.side"]["area"] < 3.15 * 6 * 6, "a 6 mm wall, its top rounded off")
        self.assertEqual(d["faces"]["fillet"]["frame"], False)
        self.assertLess(d["edges"]["top"]["length"], before)
        self.assertEqual((d["open_edges"], d["nonmanifold_edges"]), (0, 0))
        self.assertLess(d["volume"], 80 * 40 * 6)

    def test_every_name_it_gives_reads_back(self):
        """After a join, long names included: a name is spelled as `face` and `edge` read
        it, and `face` gives a frame where it says so."""
        part = s3d.box("a_very_long_part_name_that_goes_past_what_blender_keeps", 40, 20, 10)
        s3d.join(part, s3d.at(s3d.cylinder("another_very_long_part_name_past_what_blender_keeps", 10, 10), z=10))
        d = s3d.describe(part)
        self.assertTrue(any("~" in n for n in d["faces"]))
        for name, f in d["faces"].items():
            if f["frame"]:
                part.face(name)
            else:
                self.assertRaises(ValueError, part.face, name)
        for name in d["edges"]:
            part.edge(name)

    def test_a_qualified_name_only_where_it_differs(self):
        a = s3d.box("a", 20, 20, 10)
        self.assertNotIn("a.top", s3d.describe(a)["faces"])
        s3d.join(a, s3d.at(s3d.box("b", 10, 10, 10), z=10))
        faces = s3d.describe(a)["faces"]
        self.assertIn("a.top", faces)
        self.assertIn("b.top", faces)
        self.assertEqual(faces["top"]["area"], faces["a.top"]["area"] + faces["b.top"]["area"])

    def test_placement_moves_no_volume(self):
        cube = s3d.box("cube", 10, 10, 10)
        plain = s3d.describe(cube)["volume"]
        for placed in (s3d.transform(s3d.copy(cube), mirror="x"), s3d.at(s3d.copy(cube), x=12345.6, rz=30)):
            self.assertEqual(s3d.describe(placed)["volume"], plain)

    def test_no_volume_for_what_is_not_closed(self):
        cube = s3d.box("cube", 10, 10, 10)
        bm = bmesh.new()
        bm.from_mesh(cube.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.5], context="FACES_ONLY")
        bm.to_mesh(cube.data)
        bm.free()
        d = s3d.describe(cube)
        self.assertEqual((d["volume"], d["open_edges"], d["nonmanifold_edges"]), (None, 4, 0))

    def test_parts_in_the_space_of_the_whole(self):
        """A part's box is given where the described whole is, as bbox reads the whole."""
        inner = s3d.at(s3d.box("b", 10, 10, 10), x=20, rz=45)
        g = s3d.at(s3d.group("g", inner), x=100, rz=90)
        d = s3d.describe(g)
        self.assertEqual(d["parts"][0]["bbox"], d["bbox"])
        self.assertAlmostEqual(d["bbox"][0][1], s3d.bbox(g, "-Y").origin[1], places=5)

    def test_joints_at_rest_and_posed(self):
        box = s3d.box("box", 100, 60, 10)
        lid = s3d.at(s3d.box("lid", 100, 60, 4), z=10)
        s3d.joint("hinge", lid, s3d.bbox(box, "-Y+Z"), axis="+X", range=(0, 110))
        s3d.material(lid, "oak")
        chest = s3d.group("chest", box, lid)
        d = s3d.describe(chest)
        self.assertEqual((d["size"], d["volume"], d["faces"]), ([100.0, 60.0, 14.0], None, {}))
        self.assertEqual([p["name"] for p in d["parts"]], ["box", "lid"])
        hinge = {"name": "hinge", "kind": "revolute", "on": "box", "axis": [1.0, 0.0, 0.0], "range": [0.0, 110.0]}
        self.assertEqual(d["parts"][1]["joints"], [{**hinge, "value": 0.0}])
        self.assertEqual(d["parts"][1]["material"], "oak")
        s3d.pose(lid, 90)
        d = s3d.describe(chest)
        self.assertEqual(d["parts"][1]["joints"], [{**hinge, "value": 90.0}])
        self.assertAlmostEqual(d["size"][2], 70, places=5)

    def test_an_assembly_and_its_anchors(self):
        base = s3d.box("base", 40, 40, 5)
        kit = s3d.assembly("kit", base, s3d.at(s3d.box("post", 5, 5, 30), z=5), mount=base.face("top"), span=40)
        d = s3d.describe(kit)
        self.assertEqual((d["name"], d["size"], d["anchors"], [p["name"] for p in d["parts"]]),
                         ("kit", [40.0, 40.0, 35.0], ["mount", "span"], ["base", "post"]))

    def test_a_joint_whose_parent_is_gone(self):
        box = s3d.box("box", 100, 60, 10)
        lid = s3d.at(s3d.box("lid", 100, 60, 4), z=10)
        s3d.joint("hinge", lid, s3d.bbox(box, "-Y+Z"), axis="+X", range=(0, 110))
        s3d.delete(box)
        self.assertIsNone(s3d.describe(lid)["joints"][0]["on"])

    def test_an_empty_group(self):
        d = s3d.describe(s3d.at(s3d.group("e"), x=100))
        self.assertEqual((d["bbox"], d["size"], d["volume"], d["parts"]), (None, None, None, []))

    def test_not_a_part(self):
        for wrong in (s3d.rect(10, 10), s3d.Frame(), "box", None):
            with self.assertRaises(TypeError):
                s3d.describe(wrong)


if __name__ == "__main__":
    unittest.main()
