"""describe: what a part holds."""
import unittest

import bpy
import bmesh

import sand3d as s3d


class Describe(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_box(self):
        d = s3d.describe(s3d.box("box", 100, 60, 10))
        self.assertEqual((d["bbox"], d["size"], d["open_edges"]), ([[-50, -30, 0], [50, 30, 10]], [100, 60, 10], 0))
        self.assertAlmostEqual(d["volume"], 60000, places=6)
        self.assertEqual(sorted(d["faces"]), ["bottom", "side", "top"], "box.top is top: listed once")
        self.assertAlmostEqual(d["faces"]["side"], 3200, places=6)
        self.assertEqual(d["edges"], ["bottom", "side", "top"])

    def test_every_name_reads_back(self):
        """After a hole and a join, long names included."""
        part = s3d.box("a_very_long_part_name_that_goes_past_what_blender_keeps", 40, 20, 10)
        s3d.hole(part, 6, on=part.face("top"))
        s3d.join(part, s3d.at(s3d.box("b", 10, 10, 10), z=10))
        d = s3d.describe(part)
        self.assertIn("hole.side", d["faces"])
        self.assertIn("b.top", d["faces"], "after a join, top is two parts' tops")
        for name in d["edges"]:
            part.edge(name)
        part.face(next(n for n in d["faces"] if "~" in n))

    def test_volume(self):
        """Placement moves none; a mesh that is not closed has none."""
        cube = s3d.box("cube", 10, 10, 10)
        for placed in (s3d.transform(s3d.copy(cube), mirror="x"), s3d.at(s3d.copy(cube), x=12345.6, rz=30)):
            self.assertEqual(s3d.describe(placed)["volume"], s3d.describe(cube)["volume"])
        bm = bmesh.new()
        bm.from_mesh(cube.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z > 0.5], context="FACES_ONLY")
        bm.to_mesh(cube.data)
        bm.free()
        self.assertEqual((s3d.describe(cube)["open_edges"], s3d.describe(cube)["volume"]), (4, None))

    def test_parts_joints_and_pose(self):
        box = s3d.box("box", 100, 60, 10)
        lid = s3d.at(s3d.box("lid", 100, 60, 4), z=10)
        s3d.joint("hinge", lid, s3d.bbox(box, "-Y+Z"), axis="+X", range=(0, 110))
        s3d.material(lid, "oak")
        chest = s3d.at(s3d.group("chest", box, lid), x=100)
        d = s3d.describe(chest)
        self.assertEqual((d["size"], d["volume"], [p["name"] for p in d["parts"]]), ([100, 60, 14], None, ["box", "lid"]))
        self.assertEqual(d["parts"][0]["bbox"], [[50, -30, 0], [150, 30, 10]], "in the space of the whole")
        self.assertEqual(d["parts"][1]["material"], "oak")
        s3d.pose(lid, 90)
        j = s3d.describe(chest)["parts"][1]["joints"][0]
        self.assertEqual((j["name"], j["kind"], j["on"], j["range"], j["value"]), ("hinge", "revolute", "box", [0, 110], 90))

    def test_what_it_refuses_and_empties(self):
        self.assertEqual(s3d.describe(s3d.group("e"))["bbox"], None)
        for wrong in (s3d.rect(10, 10), "box", None):
            self.assertRaises(TypeError, s3d.describe, wrong)


if __name__ == "__main__":
    unittest.main()
