# sand3d

A Python library for writing 3D objects as programs. A program describes how a
shape is made, in profiles, solids, edits, frames and assemblies, and Blender
(`bpy`) computes the geometry.

```python
import sand3d as s3d

def bracket(w=80, d=40, t=6, hole_d=6.6, pitch=50):
    plate = s3d.extrude("plate", s3d.rect(w, d, r=4), t)
    s3d.hole(plate, hole_d, on=s3d.grid(2, 1, pitch, 0))
    s3d.fillet(plate, 1, edges="top")
    return plate

s3d.export(bracket(), "bracket.glb")
```

The whole API fits on one page: [docs/api.md](docs/api.md).

## Install

sand3d needs `bpy`, Blender as a Python module: `bpy==5.2.*` on Python 3.13. It also
runs under bpy 4.5 (Python 3.11), but the test suite here only covers 5.2.

```sh
pip install bpy==5.2.*
pip install "sand3d @ git+https://github.com/Sandflow-ai/sand3d@v1.5.0"
```

sand3d is not on PyPI yet.

## Writing a program

Every verb is `s3d.<verb>`. Lengths are in millimetres unless the program calls
`s3d.units(...)` before its first verb. Makers take a name and return the solid;
edits return the object they change.

The five rules a program follows:

1. **`import sand3d as s3d`, never `import *`.** Verbs are common nouns (`hole`,
   `shell`, `ring`, `text`). The prefix keeps a program from shadowing them.
2. **No `bpy` or `bmesh` in the program.** If no verb can express a shape, that
   is a missing verb, so please open an issue.
3. **Edges and faces by name, not by geometry.** A verb names the edges and faces
   it creates (`top`, `side`, `rim`, `cut`), and `fillet(ob, r, edges="top")`
   reads those names. There are no geometric selectors.
4. **Placement is relative.** Each verb returns a named object in its own frame.
   Parts are placed with `at` on frames (a face, a bounding-box side, a point on a
   path) instead of absolute matrices. Nothing reads the selection or the active
   object.
5. **Architecture first, one line per detail, every number named.** The first
   function is named after the object and assembles its parts. Each detail is one
   line that can be deleted without affecting the rest, and each dimension is a
   named parameter.

Every verb is deterministic: two fresh processes produce the same mesh. A
program can record `s3d.__version__` and `s3d.fingerprint()` (a digest of the
library source) with its output.

Example programs: [docs/examples/](docs/examples/) (bracket, lamp, rack, chair,
gate) and [programs/](programs/) (a spoon and a fork from generic verbs, and an
articulated washing machine).

## Exports

`s3d.export(root, path)` writes in metres, and the file extension selects the format:

- **`.glb`**: glTF binary at the current pose, with the materials the program set
  and the joints in node extras.
- **`.urdf`**: one link per moving part, at rest, with its meshes as OBJ in
  `<stem>_meshes/` next to the file.
- **`.usda`**: USD as text, with the URDF's links and joints in UsdGeom and
  UsdPhysics, Z up, and the materials. sand3d does not depend on USD's own
  library: convert to `.usdc` or `.usdz` with USD's tools.

Joints export limits only. Masses, frictions, efforts and drives are dynamics,
which sand3d does not describe.

## Versioning

sand3d uses semantic versioning over the surface published in
[docs/api.md](docs/api.md). From 1.5.0 on, the vocabulary only grows: existing
verbs keep their signatures and their meaning, and new verbs and parameters come
with defaults. A change that would break a program written against an earlier
1.x release requires a new major version.

## Tests

```sh
python tests/run.py            # under a Python with bpy 5.2
```

The USD export test reads files back with `usd-core` in a separate Python, set by
`SAND3D_USD_PYTHON`. Without one, it is skipped.

## License

To be decided before the public release. See [LICENSE](LICENSE).
