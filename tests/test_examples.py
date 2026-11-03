"""The worked examples of the API reference run and make closed solids."""
import importlib.util
import unittest
from pathlib import Path

import bpy
import bmesh

import sand3d as s3d
from test_regressions import shells

EXAMPLES = Path(__file__).resolve().parents[1] / "docs/examples"


class Examples(unittest.TestCase):

    def test_examples_build_closed_solids(self):
        for name in ("bracket", "lamp", "rack", "chair", "gate"):
            s3d.empty_scene()
            spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            getattr(module, name)()
            meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
            self.assertTrue(meshes)
            for ob in meshes:
                bm = bmesh.new()
                bm.from_mesh(ob.data)
                self.assertTrue(all(len(e.link_faces) == 2 for e in bm.edges), f"{name}: {ob.name} is open")
                bm.free()
                self.assertEqual(shells(ob), 1, f"{name}: {ob.name} falls apart")

    def test_version_and_fingerprint(self):
        self.assertRegex(s3d.__version__, r"^\d+\.\d+\.\d+(-alpha\.\d+)?$")   # the 1.0.0 alphas on next
        self.assertEqual(len(s3d.fingerprint()), 16)


if __name__ == "__main__":
    unittest.main()
