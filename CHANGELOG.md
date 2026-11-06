# Changelog

## 1.6.3 (2026-11-06)

### Fixes

* A GLB holds only the root's tree, in metres: the rest of the scene no longer comes
  along, and a placed or mirrored root keeps its offset and its mirror (#11)

## 1.6.2 (2026-11-04)

### Fixes

* The README's images show on PyPI

## 1.6.1 (2026-11-04)

First public release of sand3d: profiles, solids, edits, frames, joints and an
assembly layer over Blender's `bpy`, with export to GLB, URDF, USD (`.usda`,
UsdPhysics), OBJ and STL. `pip install sand3d` brings the `bpy` its Python takes:
bpy 4.5 on Python 3.11, bpy 5.2 on Python 3.13. The API is described in
[docs/api.md](docs/api.md).
