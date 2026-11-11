"""describe: what a part holds, read back as a model would before its next verb."""
import unittest

import sand3d as s3d


class Describe(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_box(self):
        d = s3d.describe(s3d.box("box", 100, 60, 10))
        self.assertEqual((d["name"], d["size"], d["open_edges"]), ("box", [100.0, 60.0, 10.0], 0))
        self.assertAlmostEqual(d["volume"], 60000, places=6)
        self.assertEqual(d["bbox"], [[-50.0, -30.0, 0.0], [50.0, 30.0, 10.0]])
        self.assertEqual(d["faces"]["box.top"]["count"], 1)
        self.assertAlmostEqual(d["faces"]["box.top"]["area"], 6000, places=6)
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
        self.assertIn("hole.side", d["faces"])
        self.assertIn("fillet", d["faces"])
        self.assertLess(d["edges"]["top"]["count"], before)
        self.assertLess(d["volume"], 80 * 40 * 6)

    def test_every_name_it_gives_reads_back(self):
        """Long names included: a name is given as `face` reads it."""
        part = s3d.box("a_very_long_part_name_that_goes_past_what_blender_keeps", 10, 10, 10)
        for name in s3d.describe(part)["faces"]:
            part.face(name)

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

    def test_not_a_part(self):
        with self.assertRaises(TypeError):
            s3d.describe(s3d.rect(10, 10))


if __name__ == "__main__":
    unittest.main()
