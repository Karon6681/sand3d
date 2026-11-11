"""Per-call hooks process the changed part, independent of scene size."""
import unittest
from unittest.mock import patch

import bpy
import sand3d as s3d
from sand3d import _core, materials


class Hooks(unittest.TestCase):
    def setUp(self):
        s3d.empty_scene()

    def test_material_work_is_linear(self):
        s3d.material(s3d.box('seed', 1, 1, 1), 'oak')
        inspected = []
        nearest = materials._nearest
        with patch.object(materials, '_nearest', side_effect=lambda ob: (inspected.append(ob), nearest(ob))[1]):
            for i in range(80):
                s3d.box(f'part{i}', 1, 1, 1)
        self.assertEqual(len(inspected), 80)

    def test_hierarchy_and_replaced_mesh_match_full_resolution(self):
        a = s3d.box('a', 3, 3, 3)
        b = s3d.box('b', 1, 1, 1)
        root = s3d.group('root', a, b)
        s3d.material(root, 'oak')
        s3d.material(a, 'steel')
        s3d.material(root, 'brass')
        s3d.group('outside', b)
        s3d.cut(a, s3d.box('tool', 1, 1, 4))
        s3d.delete(b)
        c = s3d.copy(a)
        s3d.at(c, x=5)
        before = {ob.name: list(ob.data.materials) for ob in bpy.data.objects if ob.type == 'MESH'}
        materials.resolve()
        self.assertEqual(before, {ob.name: list(ob.data.materials) for ob in bpy.data.objects if ob.type == 'MESH'})
        self.assertEqual(a.data.materials[0].name, 'steel')
