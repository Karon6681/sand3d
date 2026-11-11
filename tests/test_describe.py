"""describe: what a part holds, read back as a model would before its next verb."""
import unittest

import sand3d as s3d


class Describe(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_box(self):
        d = s3d.describe(s3d.box("box", 100, 60, 10))
        self.assertEqual((d["name"], d["size"], d["open_edges"], d["nonmanifold_edges"]),
                         ("box", [100.0, 60.0, 10.0], 0, 0))
        self.assertAlmostEqual(d["volume"], 60000, places=6)
        self.assertEqual(d["bbox"], [[-50.0, -30.0, 0.0], [50.0, 30.0, 10.0]])
        self.assertEqual(d["faces"]["box.top"]["count"], 1)
        self.assertAlmostEqual(d["faces"]["box.top"]["area"], 6000, places=6)
        self.assertEqual(d["faces"]["side"]["count"], 4)
        self.assertAlmostEqual(d["faces"]["side"]["area"], 3200, places=6)
        self.assertEqual(d["edges"]["top"], {"count": 4})
        self.assertEqual((d["joints"], d["material"], d["parts"]), ([], None, []))

    def test_names_come_and_go_with_the_verbs(self):
        """A hole adds its names; a fillet uses the edges it rounds and names its faces."""
        plate = s3d.box("plate", 80, 40, 6)
        before = s3d.describe(plate)["edges"]["top"]["count"]
        s3d.hole(plate, 6, on=plate.face("top"))
        s3d.fillet(plate, 1, edges="top")
        d = s3d.describe(plate)
        self.assertEqual(d["faces"]["hole.side"]["count"], 24)
        self.assertAlmostEqual(d["faces"]["hole.side"]["area"], sum(p.area for p in plate.data.polygons
                               if plate.data.attributes["s3d_face:hole.side"].data[p.index].value), places=6)
        self.assertIn("fillet", d["faces"])
        self.assertLess(d["edges"]["top"]["count"], before)
        self.assertEqual((d["open_edges"], d["nonmanifold_edges"]), (0, 0))
        self.assertLess(d["volume"], 80 * 40 * 6)

    def test_every_name_it_gives_reads_back(self):
        """Long names included, after a join: a name is spelled as `face` and `edge` read it."""
        part = s3d.box("a_very_long_part_name_that_goes_past_what_blender_keeps", 40, 20, 10)
        s3d.join(part, s3d.at(s3d.box("another_very_long_part_name_past_what_blender_keeps", 10, 10, 10), z=10))
        d = s3d.describe(part)
        for name in d["faces"]:
            if any(c in name for c in ("top", "bottom")):
                part.face(name)
        for name in d["edges"]:
            part.edge(name)

    def test_placement_moves_no_volume(self):
        cube = s3d.box("cube", 10, 10, 10)
        plain = s3d.describe(cube)["volume"]
        for placed in (s3d.transform(s3d.copy(cube), mirror="x"), s3d.at(s3d.copy(cube), x=12345.6, rz=30)):
            self.assertEqual(s3d.describe(placed)["volume"], plain)

    def test_parts_in_the_space_of_the_whole(self):
        """A part's box is given where the described whole is, as bbox reads the whole."""
        inner = s3d.at(s3d.box("b", 10, 10, 10), x=20, rz=45)
        g = s3d.at(s3d.group("g", inner), x=100, rz=90)
        d = s3d.describe(g)
        self.assertEqual(d["parts"][0]["bbox"], d["bbox"])
        self.assertAlmostEqual(d["bbox"][0][1], s3d.bbox(g, "-Y").origin[1], places=5)

    def test_joints_material_and_parts(self):
        box = s3d.box("box", 100, 60, 10)
        lid = s3d.at(s3d.box("lid", 100, 60, 4), z=10)
        s3d.joint("hinge", lid, s3d.bbox(box, "-Y+Z"), axis="+X", range=(0, 110))
        s3d.material(lid, "oak")
        d = s3d.describe(s3d.group("chest", box, lid))
        self.assertEqual(d["size"], [100.0, 60.0, 14.0])
        self.assertNotIn("volume", d)
        self.assertEqual([p["name"] for p in d["parts"]], ["box", "lid"])
        self.assertEqual(d["parts"][1]["joints"], [{"name": "hinge", "kind": "revolute", "on": "box"}])
        self.assertEqual(d["parts"][1]["material"], "oak")

    def test_an_assembly(self):
        kit = s3d.assembly("kit", s3d.box("base", 40, 40, 5), s3d.at(s3d.box("post", 5, 5, 30), z=5))
        d = s3d.describe(kit)
        self.assertEqual((d["name"], d["size"], [p["name"] for p in d["parts"]]),
                         ("kit", [40.0, 40.0, 35.0], ["base", "post"]))

    def test_a_joint_whose_parent_is_gone(self):
        box = s3d.box("box", 100, 60, 10)
        lid = s3d.at(s3d.box("lid", 100, 60, 4), z=10)
        s3d.joint("hinge", lid, s3d.bbox(box, "-Y+Z"), axis="+X", range=(0, 110))
        s3d.delete(box)
        self.assertEqual(s3d.describe(lid)["joints"], [{"name": "hinge", "kind": "revolute", "on": None}])

    def test_an_empty_group(self):
        d = s3d.describe(s3d.at(s3d.group("e"), x=100))
        self.assertEqual((d["bbox"], d["size"], d["parts"]), (None, None, []))

    def test_not_a_part(self):
        for wrong in (s3d.rect(10, 10), s3d.Frame(), "box", None):
            with self.assertRaises(TypeError):
                s3d.describe(wrong)


if __name__ == "__main__":
    unittest.main()
