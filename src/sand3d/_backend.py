"""The one place sand3d meets the 3D application it runs in.

Every call into Blender (`bpy`, `bmesh`, `mathutils`) is a method or an attribute of
the backend `B`. Another application runs sand3d through a class with the same methods:
`SAND3D_BACKEND="package.module:Class"` in the environment, before `sand3d` is
imported, names it; Blender's is the default. The rest of the library reads `B` at
import, so the backend is chosen once per process.

What a backend hands the library, and what the library does with it:

Math. `Matrix`, `Vector`, `Euler` and `KDTree` are classes with mathutils' interface,
in single precision as mathutils is (the digests depend on its rounding). The library
uses: `Vector(seq)`, `+ - * / @`, unary `-`, indexing, `.x .y .z .xyz`, `.length`,
`.length_squared`, `.dot`, `.cross`, `.normalized()`, `.normalize()`, `.angle(v, fallback)`,
`.lerp(v, t)`, `.orthogonal()`, `.rotation_difference(v) @ v`, `.copy()`;
`Matrix(rows)`, `Matrix.Identity(n)`, `Matrix.Translation(v)`, `Matrix.Rotation(angle, n,
axis)`, `Matrix.Scale(s, n)`, `Matrix.Diagonal(v)`, `m[r][c]` (read and write), `@`,
`.inverted()`, `.to_3x3()`, `.to_4x4()`, `.to_translation()`, `.translation`, `.col[i]`,
`.determinant()`, `==`; `Euler((x, y, z)).to_matrix()`; `KDTree(n)` with `insert(co, index)`,
`balance()`, `find(co) -> (co, index, dist)`, `find_range(co, r) -> [(co, index, dist)]`.

Objects. What the verbs return is the backend's object, read and written by the library
through these attributes only: `name` (read, write), `type` ("MESH" or "EMPTY"),
`parent` (read, write; None at the top), `children`, `children_recursive`,
`matrix_basis` (a 4x4 `Matrix`, read and write: the placement in the parent),
`matrix_parent_inverse` (read, write), `data` (its mesh, or None: opaque to the library,
handed back to the mesh methods below), and per-object custom values as a mapping:
`ob[key]`, `ob[key] = value`, `ob.get(key, default)`, `key in ob`, `del ob[key]`,
`ob.keys()`, `ob.items()`. A value is a str, a number, a bool, a list or dict of those,
or another object (a reference, which stays the same object). Objects are hashable and
compare by identity. A material made by `new_material` takes custom values the same way.
`install_object_methods` adds the library's own methods (`ob.face`, `ob.edge`) to them.

Edit meshes. A mesh being built or edited (`edit_mesh`) is a half-edge mesh with
BMesh's interface, as far as the library walks it:
  `bm.verts`, `bm.edges`, `bm.faces`: sequences (iteration, `len`, `seq[i]` once
  `seq.ensure_lookup_table()` ran), `seq.index_update()` numbering the elements in order;
  `bm.verts.new(co)`, `bm.faces.new(verts)` (raises ValueError if the face exists);
  `bm.normal_update()`, `bm.free()`.
  vertex: `co` (a `Vector`, writable), `index`, `normal`, `link_faces`, `link_edges`.
  edge: `verts`, `link_faces`, `index`, `smooth` (bool, writable), `is_valid`,
  `calc_length()`, `other_vert(v)`.
  face: `verts` (in winding order), `edges`, `normal`, `index`, `smooth` (writable),
  `is_valid`, `calc_area()`.
  `element[layer]` reads and writes a per-element value of a layer (`new_layer`).
Faces and edges carry named layers: bool tags (`s3d_face:<name>`, `s3d_edge:<name>`), and
a few int or float scratch layers. Adding or removing a layer invalidates the Python
references to elements: the library re-reads them by index after one.
"""
import atexit
import importlib
import os


class BlenderBackend:
    """sand3d inside Blender, or `bpy` as a Python module (4.5 and 5.x)."""

    name = "blender"

    def __init__(self):
        import bpy  # before bmesh, which the bpy module only provides once loaded
        import bmesh
        import mathutils
        from mathutils import bvhtree, geometry, kdtree
        self._bpy, self._bmesh, self._geometry, self._bvh = bpy, bmesh, geometry, bvhtree.BVHTree
        self.Matrix, self.Vector, self.Euler = mathutils.Matrix, mathutils.Vector, mathutils.Euler
        self.KDTree = kdtree.KDTree

    # ---------------------------------------------------------------- 2D geometry

    def delaunay_2d_cdt(self, points, edges, output_type, eps):
        """Constrained Delaunay triangulation of 2D `points` (Vectors) with the constraint
        `edges` (index pairs), as `mathutils.geometry.delaunay_2d_cdt(points, edges, [],
        output_type, eps)`: (verts, edges, faces, orig_verts, orig_edges, orig_faces).
        `output_type` 0 keeps the triangles inside the constraint loops only, 1 adds the
        faces of the convex hull too. The library reads `faces` and `orig_verts` (for each
        output vertex, the input indices it came from; empty for a new one), and `verts`."""
        return self._geometry.delaunay_2d_cdt(points, edges, [], output_type, eps)

    def intersect_line_line_2d(self, a, b, c, d):
        """Where the segments a-b and c-d (2D Vectors) cross, or None."""
        return self._geometry.intersect_line_line_2d(a, b, c, d)

    # ---------------------------------------------------------------- objects and the scene

    def is_object(self, value):
        """Whether `value` is an object of the scene (or one removed from it)."""
        return isinstance(value, self._bpy.types.Object)

    def is_data(self, value):
        """Whether `value` is any of the backend's data (an object, a mesh, a material)."""
        return isinstance(value, self._bpy.types.ID)

    def is_removed(self, value):
        """Whether `value`, an object of the backend's, was removed: its Python handle
        outlived it."""
        if not isinstance(value, self._bpy.types.ID):
            return False
        try:
            value.name
        except ReferenceError:
            return True
        return False

    def key(self, ob):
        """A hashable identity of `ob` that stays while it lives."""
        return ob.as_pointer()

    def link(self, ob):
        """Put `ob` in the scene, where the exporter sees it."""
        self._bpy.context.scene.collection.objects.link(ob)
        return ob

    def new_object(self, name, mesh=None, collection=None):
        """A new object named `name` (the backend may suffix it to keep names unique):
        a mesh object holding `mesh`, or a null (an empty) if `mesh` is None; at the
        identity placement, with no parent, in `collection` (from `new_collection`) or
        else in the scene."""
        ob = self._bpy.data.objects.new(name, mesh)
        if collection is not None:
            collection.objects.link(ob)
        else:
            self.link(ob)
        return ob

    def copy_object(self, ob):
        """A new object like `ob` in the scene: placement, parent, custom values, and its
        own copy of the mesh, sharing nothing with `ob`'s."""
        twin = ob.copy()                   # Blender shares the mesh: copy it
        if ob.data is not None:
            twin.data = ob.data.copy()
        return self.link(twin)

    def remove_object(self, ob):
        """Remove `ob` from the scene, and its mesh if nothing else uses it."""
        mesh = ob.data if ob.type == "MESH" else None
        self._bpy.data.objects.remove(ob)
        if mesh is not None and mesh.users == 0:
            self._bpy.data.meshes.remove(mesh)

    def objects(self):
        """Every object, in the backend's own order (Blender's: by name)."""
        return list(self._bpy.data.objects)

    def scene_objects(self):
        """Every object of the scene, in the scene's order."""
        return list(self._bpy.context.scene.objects)

    def reset_scene(self):
        """An empty scene: no object, mesh, material or collection left."""
        self._bpy.ops.wm.read_factory_settings(use_empty=True)

    def value(self, ob, key):
        """`ob`'s custom value `key` as plain Python (dicts and lists), or None."""
        value = ob.get(key)
        if value is not None and hasattr(value, "to_dict"):
            return value.to_dict()
        return value

    def install_object_methods(self, methods):
        """Make each of `methods` ({name: function(ob, ...)}) a method of every object:
        `ob.face("top")` calls `methods["face"](ob, "top")`."""
        types = self._bpy.types
        for name, fn in methods.items():
            setattr(types.Object, name, fn)
        # bpy 4.5 never exits while a Python attribute sits on an RNA type: take them off first.
        atexit.register(lambda: [delattr(types.Object, a) for a in methods if hasattr(types.Object, a)])

    def placement(self, ob):
        """`ob`'s placement as the backend holds it, as plain values that bring it back to
        the bit (`set_placement`): Blender keeps location, rotation and scale, and its
        `matrix_basis` is made from them, so a matrix set and read again may differ. Its
        "matrix" entry is `matrix_basis` (16 floats, row by row), "mode" the rotation mode."""
        return {"matrix": [float(ob.matrix_basis[r][c]) for r in range(4) for c in range(4)],
                "mode": ob.rotation_mode, "location": list(ob.location), "euler": list(ob.rotation_euler),
                "quaternion": list(ob.rotation_quaternion), "axis_angle": list(ob.rotation_axis_angle),
                "scale": list(ob.scale)}

    def set_placement(self, ob, kept):
        """`ob` placed by what `placement` returned."""
        ob.rotation_mode = kept["mode"]
        ob.location = kept["location"]
        ob.rotation_euler = kept["euler"]
        ob.rotation_quaternion = kept["quaternion"]
        ob.rotation_axis_angle = kept["axis_angle"]
        ob.scale = kept["scale"]

    def placement_of(self, matrix, mode):
        """What `placement` would return for an object whose rotation mode is `mode`
        (from an earlier `placement`) once `matrix` is set as its `matrix_basis`."""
        scratch = self._bpy.data.objects.new("s3d~scratch", None)   # Blender's own decomposition
        try:
            scratch.rotation_mode = mode
            scratch.matrix_basis = matrix
            return self.placement(scratch)
        finally:
            self._bpy.data.objects.remove(scratch)

    # ---------------------------------------------------------------- collections (export staging)

    def new_collection(self, name):
        """A new, empty collection of objects in the scene: what the GLB export writes."""
        coll = self._bpy.data.collections.new(name)
        self._bpy.context.scene.collection.children.link(coll)
        return coll

    def collection_objects(self, coll):
        """The objects in `coll`."""
        return list(coll.objects)

    def remove_collection(self, coll):
        """Remove `coll` (not the objects in it)."""
        self._bpy.data.collections.remove(coll)

    # ---------------------------------------------------------------- reading a mesh

    def mesh_coords(self, me):
        """`me`'s vertex coordinates as one flat list [x0, y0, z0, x1, ...] of floats
        (single precision values)."""
        co = [0.0] * (3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
        return co

    def mesh_points(self, me):
        """`me`'s vertex coordinates as `Vector`s, in vertex order."""
        return [v.co for v in me.vertices]

    def mesh_faces(self, me):
        """`me`'s faces as tuples of vertex indices, in face order and winding order."""
        return [tuple(p.vertices) for p in me.polygons]

    def mesh_edges(self, me):
        """`me`'s edges as (vertex, vertex) pairs, in edge order."""
        return [tuple(e.vertices) for e in me.edges]

    def mesh_counts(self, me):
        """(number of vertices, number of faces) of `me`."""
        return len(me.vertices), len(me.polygons)

    def mesh_polygons(self, me):
        """`me`'s faces as records with `index`, `vertices` (indices, winding order),
        `normal` (a `Vector`), `center` (a `Vector`, the mean of its corners) and `area`:
        Blender's own values, computed in single precision."""
        return list(me.polygons)

    def mesh_attributes(self, me):
        """{name: domain} of `me`'s named per-element values ("POINT", "EDGE", "FACE",
        "CORNER"), in the backend's order: at least every tag (`s3d_face:`, `s3d_edge:`)."""
        return {a.name: a.domain for a in me.attributes}

    def mesh_attribute_values(self, me, name):
        """The values of `me`'s attribute `name`, one per element of its domain."""
        return [d.value for d in me.attributes[name].data]

    def mesh_triangles(self, me, normals=False):
        """`me`'s faces cut into triangles, as triples of vertex indices, in the winding of
        their face: Blender's own triangulation (`loop_triangles`), which a concave face
        needs. With `normals`, each triangle comes as (vertices, [normal at each corner]),
        the corner normals as the backend shades them (smooth or sharp)."""
        me.calc_loop_triangles()
        if not normals:
            return [tuple(t.vertices) for t in me.loop_triangles]
        corner = [0.0] * (3 * len(me.loops))
        me.corner_normals.foreach_get("vector", corner)
        return [(list(t.vertices), [corner[3 * l:3 * l + 3] for l in t.loops]) for t in me.loop_triangles]

    # ---------------------------------------------------------------- changing a mesh

    def copy_mesh(self, me):
        """A new mesh like `me`, attributes and all, sharing nothing with it."""
        return me.copy()

    def remove_mesh(self, me):
        """Remove the mesh `me`, which no object uses."""
        self._bpy.data.meshes.remove(me)

    def transform_mesh(self, me, matrix):
        """Move every vertex of `me` by `matrix` (a 4x4 `Matrix`)."""
        me.transform(matrix)

    def flip_mesh_normals(self, me):
        """Reverse the winding of every face of `me`."""
        me.flip_normals()

    def set_mesh_coords(self, me, co):
        """Set `me`'s vertex coordinates from one flat list, as `mesh_coords` gives."""
        me.vertices.foreach_set("co", co)
        me.update()

    def remove_mesh_attribute(self, me, name):
        """Remove `me`'s attribute `name`."""
        me.attributes.remove(me.attributes[name])  # by name: removing one invalidates the others (bpy 4.5)

    def add_face_attribute(self, me, name, values=None):
        """A per-face integer attribute `name` on `me`: `values`, or 0 on every face."""
        att = me.attributes.new(name, "INT", "FACE")
        if values is not None:
            att.data.foreach_set("value", values)

    def set_object_mesh(self, ob, bm):
        """`bm` (an edit mesh, freed) as `ob`'s new mesh, returned; the old mesh is removed
        if nothing else uses it. Normals are not recomputed."""
        old = ob.data
        mesh = self._bpy.data.meshes.new(old.name)
        bm.to_mesh(mesh)
        bm.free()
        ob.data = mesh
        if old.users == 0:
            self._bpy.data.meshes.remove(old)
        return mesh

    def set_mesh_name(self, me, name):
        """Name the mesh `me` (the GLB carries it)."""
        me.name = name

    def write_mesh(self, bm, me):
        """Write the edit mesh `bm` into the existing mesh `me`, in place (`bm` stays)."""
        bm.to_mesh(me)

    # ---------------------------------------------------------------- materials

    def materials(self):
        """Every material made (`new_material`)."""
        return list(self._bpy.data.materials)

    def new_material(self, name, rgba, v):
        """A material named `name`: base and emission colour `rgba` (linear), and from
        the dict `v` roughness, metallic, opacity (alpha, blended below 1), transmission,
        coat (coat roughness 0.05 when there is a coat), sheen, emission (strength), as a
        principled / glTF metallic-roughness surface. It takes custom values as an object."""
        mat = self._bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        for socket, value in (("Base Color", rgba), ("Roughness", v["roughness"]), ("Metallic", v["metallic"]),
                              ("Alpha", v["opacity"]), ("Transmission Weight", v["transmission"]),
                              ("Coat Weight", v["coat"]), ("Sheen Weight", v["sheen"]),
                              ("Emission Color", rgba), ("Emission Strength", v["emission"])):
            bsdf.inputs[socket].default_value = value
        if v["coat"]:
            bsdf.inputs["Coat Roughness"].default_value = 0.05
        if v["opacity"] < 1:
            mat.surface_render_method = "BLENDED"
        mat.diffuse_color = (*rgba[:3], v["opacity"])       # what Workbench and the viewport show
        mat.metallic, mat.roughness = v["metallic"], v["roughness"]
        return mat

    def mesh_materials(self, me):
        """The materials of `me`'s slots, in order (None for an empty slot)."""
        return list(me.materials)

    def set_mesh_materials(self, me, mats):
        """`me`'s slots: exactly `mats`, in order ([] for none)."""
        me.materials.clear()
        for mat in mats:
            me.materials.append(mat)

    def zero_material_indices(self, me):
        """Every face of `me` on its first material slot."""
        if any(p.material_index for p in me.polygons):
            me.polygons.foreach_set("material_index", [0] * len(me.polygons))

    # ---------------------------------------------------------------- edit meshes

    def edit_mesh(self, me=None):
        """A new edit mesh: empty, or a copy of the mesh `me` with all its attributes as layers."""
        bm = self._bmesh.new()
        if me is not None:
            bm.from_mesh(me)
        return bm

    def mesh_object(self, name, bm, recalc=True):
        """A new mesh object `name` in the scene from the edit mesh `bm` (freed), its mesh
        named `name` too; with `recalc`, its face normals made consistent and pointing out
        first (`recalc_normals`). Layers become the mesh's attributes; face and edge
        `smooth` flags its shading."""
        if recalc:
            self.recalc_normals(bm)
        mesh = self._bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        return self.new_object(name, mesh)

    def layers(self, bm, domain, kind):
        """{name: layer} of `bm`'s layers on `domain` ("verts", "edges", "faces") of `kind`
        ("bool", "int", "float"), in the order they were made."""
        return {lay.name: lay for lay in getattr(getattr(bm, domain).layers, kind).values()}

    def layer(self, bm, domain, kind, name):
        """`bm`'s layer `name` on `domain` of `kind`, or None."""
        return getattr(getattr(bm, domain).layers, kind).get(name)

    def new_layer(self, bm, domain, kind, name):
        """A new layer `name` on `domain` of `kind`: False, 0 or 0.0 on every element."""
        return getattr(getattr(bm, domain).layers, kind).new(name)

    def remove_layer(self, bm, domain, kind, name):
        """Remove `bm`'s layer `name` on `domain` of `kind`."""
        layers = getattr(getattr(bm, domain).layers, kind)
        layers.remove(layers[name])

    def recalc_normals(self, bm):
        """Make the winding of `bm`'s faces consistent across each connected shell, each
        shell's normals pointing out (`bmesh.ops.recalc_face_normals`)."""
        self._bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    def dissolve_edges(self, bm, edges):
        """Remove `edges` of `bm`, merging the two faces of each into one; their vertices
        stay, and a face is never split (`bmesh.ops.dissolve_edges`, use_verts and
        use_face_split False)."""
        self._bmesh.ops.dissolve_edges(bm, edges=edges, use_verts=False, use_face_split=False)

    def weld(self, bm, dist):
        """Merge the vertices of `bm` closer than `dist` (`bmesh.ops.remove_doubles` on all
        of them): faces and edges left degenerate go."""
        self._bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)

    def dissolve_degenerate(self, bm, dist):
        """Collapse the edges of `bm` shorter than `dist`, and the faces of no area they
        leave (`bmesh.ops.dissolve_degenerate` on all edges)."""
        self._bmesh.ops.dissolve_degenerate(bm, dist=dist, edges=bm.edges)

    def delete_faces(self, bm, faces):
        """Remove `faces` of `bm`, leaving their edges and vertices."""
        self._bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")

    def scale_mesh(self, bm, s):
        """Scale every vertex of `bm` by `s` about the origin."""
        self._bmesh.ops.scale(bm, vec=(s, s, s), verts=bm.verts)

    def sort_mesh(self, bm):
        """Renumber `bm`: vertices by position, edges and faces by their vertices' new
        indices, so the order no longer depends on how they were made (bevel and the
        boolean make them in an order that changes with the run or the thread count).
        Only where nothing holds an element."""
        def by(seq, key):
            rank = {x: i for i, x in enumerate(sorted(seq, key=key))}
            seq.sort(key=rank.__getitem__)
            seq.index_update()
        by(bm.verts, lambda v: tuple(v.co))
        by(bm.edges, lambda e: sorted(v.index for v in e.verts))
        by(bm.faces, lambda f: [v.index for v in f.verts])

    def bisect(self, bm, faces, co, no, dist):
        """Cut `faces` of `bm` (a set: with their edges and vertices) by the plane through
        `co` with normal `no`, `dist` the tolerance of a vertex on it: faces split, nothing
        removed (`bmesh.ops.bisect_plane`)."""
        geom = list(faces) + list({e for f in faces for e in f.edges}) + list({v for f in faces for v in f.verts})
        self._bmesh.ops.bisect_plane(bm, geom=geom, dist=dist, plane_co=co, plane_no=no)

    def bevel(self, bm, edges, offset, segments):
        """Bevel `edges` of `bm` by `offset` along each face (offset type OFFSET), in
        `segments` pieces with a round profile (0.5, a circular arc), the overlap clamped
        and the new vertices slid along the edge loops (`bmesh.ops.bevel`, affect EDGES,
        clamp_overlap, loop_slide). Returns the new faces."""
        out = self._bmesh.ops.bevel(bm, geom=edges, offset=offset, offset_type="OFFSET", segments=segments,
                                    profile=0.5, affect="EDGES", clamp_overlap=True, loop_slide=True)
        return out["faces"]

    def bevel_weighted(self, bm, picks, segments, size):
        """Bevel every picked edge [(index, setback)] of `bm` in one operation, each at its
        own setback, with `bevel`'s options, the mesh scaled by `size` for the bevel (its
        tolerances are in its own units) and back: (a new edit mesh, its new faces). Frees
        `bm`, drops the weight layer used. Blender's bevel modifier scales one width by an
        edge weight, so each setback is the largest times its share; its new faces are told
        apart by the material index it gives them, put back to 0."""
        bpy, Matrix = self._bpy, self.Matrix
        top = max(s for _, s in picks)
        me = bpy.data.meshes.new("_s3d_bevel")
        bm.to_mesh(me)
        bm.free()
        if size != 1.0:
            me.transform(Matrix.Scale(size, 4))
        weight = me.attributes.new("s3d~weight", "FLOAT", "EDGE")
        for i, s in picks:
            weight.data[i].value = s / top
        for p in me.polygons:
            p.material_index = 0
        me.materials.append(None)
        me.materials.append(None)
        temp = self.new_object("_s3d_bevel", me)
        mod = temp.modifiers.new("bevel", "BEVEL")
        mod.limit_method, mod.edge_weight, mod.affect = "WEIGHT", "s3d~weight", "EDGES"
        mod.width, mod.offset_type, mod.segments, mod.profile = top * size, "OFFSET", segments, 0.5
        mod.use_clamp_overlap, mod.loop_slide, mod.material = True, True, 1
        depsgraph = bpy.context.evaluated_depsgraph_get()
        result = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                                 depsgraph=depsgraph)
        bpy.data.objects.remove(temp)
        bpy.data.meshes.remove(me)
        out = self._bmesh.new()
        out.from_mesh(result)
        bpy.data.meshes.remove(result)
        if size != 1.0:
            self._bmesh.ops.scale(out, vec=(1 / size,) * 3, verts=out.verts)
        lay = out.edges.layers.float.get("s3d~weight")
        if lay is not None:
            out.edges.layers.float.remove(lay)
        out.faces.ensure_lookup_table()
        faces = [f for f in out.faces if f.material_index == 1]
        for f in out.faces:
            f.material_index = 0
        return out, faces

    def boolean(self, base, tools, operation, self_intersect):
        """One exact boolean of the mesh `base` with the meshes `tools` (all in one
        space): `operation` "UNION", "DIFFERENCE" or "INTERSECT", the tools taken
        together; with `self_intersect`, each operand may cross itself or overlap another
        and is fused first. Blender's EXACT solver. Returns the result as an edit mesh,
        every attribute carried over from the faces and edges it came from (an `s3d~operand`
        per-face int says which operand). Removes the meshes given."""
        bpy = self._bpy
        coll = bpy.data.collections.new("_s3d_tools")
        bpy.context.scene.collection.children.link(coll)
        ob = self.new_object("_s3d_base", base)
        objects = []
        for me in tools:
            t = bpy.data.objects.new("_s3d_tool", me)
            coll.objects.link(t)
            objects.append(t)
        mod = ob.modifiers.new("boolean", "BOOLEAN")
        mod.operation, mod.solver = operation, "EXACT"
        mod.use_self = self_intersect
        if len(objects) == 1:
            mod.operand_type, mod.object = "OBJECT", objects[0]
        else:
            mod.operand_type, mod.collection = "COLLECTION", coll
        depsgraph = bpy.context.evaluated_depsgraph_get()
        result = bpy.data.meshes.new_from_object(ob.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                                 depsgraph=depsgraph)
        for t in objects:
            bpy.data.objects.remove(t)
        bpy.data.objects.remove(ob)
        bpy.data.collections.remove(coll)
        for me in [base, *tools]:
            if me.users == 0:
                bpy.data.meshes.remove(me)
        bm = self._bmesh.new()
        bm.from_mesh(result)
        bpy.data.meshes.remove(result)
        return bm

    # ---------------------------------------------------------------- queries

    def bvh(self, bm):
        """A BVH over the faces of the edit mesh `bm`, with Blender's `BVHTree` queries:
        `find_nearest(co) -> (location, normal, face index, distance)`,
        `ray_cast(origin, direction) -> (location, normal, face index, distance)`
        (all None on a miss; the normal is the face's), and
        `find_nearest_range(co, r) -> [(location, normal, face index, distance)]`."""
        return self._bvh.FromBMesh(bm)

    def bvh_of_mesh(self, me, matrix, size):
        """`bvh` of the mesh `me` moved by `matrix`, then scaled by `size` about the origin."""
        bm = self._bmesh.new()
        bm.from_mesh(me)
        bm.transform(matrix)
        if size != 1.0:
            self._bmesh.ops.scale(bm, vec=(size,) * 3, verts=bm.verts)
        tree = self._bvh.FromBMesh(bm)
        bm.free()
        return tree

    # ---------------------------------------------------------------- text

    def text_outline(self, s, size, font=None):
        """The text `s` in letters `size` high (the font's em) on the XY plane, filled:
        (points [(x, y)], faces [vertex indices]), each face wound anticlockwise seen from
        +Z. Curves sampled 6 steps per segment; `font` the path of a .ttf/.otf file, or the
        backend's own font."""
        bpy = self._bpy
        curve = bpy.data.curves.new("_text", "FONT")
        curve.body = s
        curve.size = size
        curve.resolution_u = 6
        if font:
            curve.font = bpy.data.fonts.load(str(font), check_existing=True)
        ob = self.new_object("_text", curve)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(ob.evaluated_get(depsgraph), depsgraph=depsgraph)
        bpy.data.objects.remove(ob)
        bpy.data.curves.remove(curve)
        co = [(v.co.x, v.co.y) for v in mesh.vertices]
        faces = []
        for poly_ in mesh.polygons:
            vs = list(poly_.vertices)
            faces.append(vs[::-1] if poly_.normal.z < 0 else vs)
        bpy.data.meshes.remove(mesh)
        return co, faces

    # ---------------------------------------------------------------- glTF

    def lend_meshes(self):
        """Before a GLB export: give each mesh object of the scene a mesh of its own for
        the exporter, a copy rewritten through bmesh, which shares no array with anything.
        The exporter runs `Mesh.validate()` on what it exports; on the program's own mesh it
        would edit the scene, and under bpy 4.5 also the copies sharing that mesh's arrays.
        The copy takes the mesh's name, which the GLB carries. Returns what
        `return_meshes` needs. (A backend whose exporter edits nothing does nothing.)"""
        objects, copies = [], {}
        for ob in self._bpy.context.scene.objects:
            if ob.type != "MESH":
                continue
            mesh = ob.data
            if mesh not in copies:
                name, own = mesh.name, mesh.copy()
                bm = self._bmesh.new()
                bm.from_mesh(own)
                bm.to_mesh(own)
                bm.free()
                mesh.name = "s3d~lent"
                own.name = name
                copies[mesh] = (own, name)
            objects.append((ob, mesh))
            ob.data = copies[mesh][0]
        return objects, copies

    def return_meshes(self, lent):
        """Undo `lend_meshes`."""
        objects, copies = lent
        for ob, mesh in objects:
            ob.data = mesh
        for mesh, (own, name) in copies.items():
            self._bpy.data.meshes.remove(own)
            mesh.name = name

    def export_glb(self, path, coll, name, dressed, material_key):
        """Write the objects of `coll` (and what hangs on them) as a binary glTF at `path`:
        its scene named `name`, each node's custom values in its extras (save the custom
        value `material_key` of materials, which is taken off meanwhile), the materials if
        `dressed` (none otherwise), +Y up as glTF wants."""
        bpy = self._bpy
        tags = {m: m.pop(material_key) for m in bpy.data.materials if material_key in m}   # extras: joints only
        clash = bpy.data.collections.get(name)  # a collection of the program's, lent its name meanwhile
        try:
            if clash is not None:
                clash.name = "s3d~export~lent"
            kept = coll.name
            coll.name = name
            try:
                bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", collection=coll.name,
                                          export_extras=True,
                                          export_materials="EXPORT" if dressed else "NONE")
            finally:
                coll.name = kept
        finally:
            if clash is not None:
                clash.name = name
            for m, tag in tags.items():
                m[material_key] = tag


def _load():
    spec = os.environ.get("SAND3D_BACKEND")
    if not spec:
        return BlenderBackend()
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr or "Backend")()


#: The backend every module of the library calls.
B = _load()
