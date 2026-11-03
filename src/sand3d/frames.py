"""Frames and placement: where things are built and where they sit.

A frame is an origin and three axes, in the frame of the space an object lives
in (the world, for an object with no parent). `ob.face("top")` is the frame of a
named face, `ob.face("+Z")` of the face furthest along +Z; distributors (`grid`,
`ring`, `along`, `snap`) return lists of frames, which move, turn and offset as
one frame does; `at` puts an object, a profile or a path on one, or a copy on each. Axes follow CadQuery's convention: on a face whose normal is
±Z, X is world +X; on any other, X is Z × normal — so a wall facing -Y has X to
the right and Y up.

A frame read on a part — `ob.face`, `bbox`, an assembly's anchor — remembers the part
(`part`) and where the part stood when it was read, through `.move`, `.turn` and
`.offset`: a joint takes its pivot from it, wherever the part has gone since.
"""
import atexit
import math
import re

import bpy
import bmesh
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import _core
from .profile import Path, Placed, Profile, _as_path


def _clean(v):
    """A unit vector without the single-precision dust meshes leave on axis-aligned
    normals: a +Z face read as (0, 1e-8, 1) would turn a frame's X anywhere."""
    v = v.normalized()
    v = Vector([0.0 if abs(c) < 1e-6 else c for c in v])
    return v.normalized()


class Frame:
    """A place to build on.

    A place to build on: an `origin`, a `normal` (its Z) and an `up` (its Y);
    without `up`, X is Z × normal (+X when the normal is ±Z)."""

    def __init__(self, origin=(0, 0, 0), normal=(0, 0, 1), up=None):
        for what, v in (("an origin", origin), ("a normal", normal), ("an up", up)):
            if v is not None and len(v) != 3:
                raise ValueError(f"Frame: {what} is (x, y, z), not {tuple(v)}")
        z = _clean(Vector(normal))
        if up is None:
            x = Vector((0, 0, 1)).cross(z)
            x = x.normalized() if x.length > 1e-6 else Vector((1, 0, 0))
            y = z.cross(x)
        else:
            y = _clean(Vector(up))
            y = y - z * y.dot(z)
            if y.length < 1e-9:
                raise ValueError(f"Frame: up {tuple(up)} is along the normal {tuple(normal)}")
            y.normalize()
        self.origin, self.x, self.y, self.z = Vector(origin), y.cross(z), y, z

    #: The part the frame was read on, and the part's placement then (`_basis`, which maps
    #: the part's own space to the one the frame is in; `_seen`, its place in the world):
    #: None for a frame built by hand.
    part = None
    _basis = None
    _seen = None

    @property
    def normal(self):
        return self.z

    def _read_on(self, part, basis, seen=None):
        """This frame, remembered as read on `part` placed by `basis`, in the world at `seen`
        (where the part is now, by default)."""
        self.part, self._basis = part, Matrix(basis)
        self._seen = Matrix(seen) if seen is not None else _core.world(part)
        return self

    def _kept(self, frame):
        """`frame`, made from this one: it remembers the same part."""
        return frame._read_on(self.part, self._basis, self._seen) if self.part is not None else frame

    def _in_part(self):
        """This frame in its part's own space: where it stays when the part moves."""
        return self._basis.inverted() @ self.matrix

    def __iter__(self):
        raise TypeError("a Frame is not a point: use f.origin (f.origin.x, f.origin.y, f.origin.z)")

    @property
    def matrix(self):
        m = Matrix.Identity(4)
        for col, v in enumerate((self.x, self.y, self.z, self.origin)):
            for row in range(3):
                m[row][col] = v[row]
        return m

    def offset(self, d):
        """Along the normal.

        The same frame moved `d` along its normal."""
        return self._kept(Frame(self.origin + self.z * d, self.z, self.y))

    def turn(self, rx=0, ry=0, rz=0):
        """About own axes, degrees.

        The same frame turned about its own X, then Y, then Z, in degrees."""
        r = Euler((math.radians(rx), math.radians(ry), math.radians(rz))).to_matrix()
        axes = self.matrix.to_3x3() @ r
        return self._kept(Frame(self.origin, axes.col[2], axes.col[1]))

    def move(self, x=0, y=0, z=0):
        """Along own axes.

        The same frame moved along its own axes."""
        return self._kept(Frame(self.point(x, y, z), self.z, self.y))

    def point(self, x=0, y=0, z=0):
        """In parent's frame.

        Where (x, y, z) of this frame lies in its parent's."""
        return self.origin + self.x * x + self.y * y + self.z * z

    def __repr__(self):
        r = lambda v: tuple(round(c, 6) + 0.0 for c in v)  # noqa: E731
        return f"Frame(origin={r(self.origin)}, normal={r(self.z)}, up={r(self.y)})"


class Frames(list):
    """Frames, as the distributors give them: the list moves, turns and offsets as one
    frame does, each frame along its own axes — on a ring, X points outward.
    `point` stays a frame's. Sliced, added to frames or repeated, it stays a `Frames`."""

    def offset(self, d):
        return Frames(f.offset(d) for f in self)

    def turn(self, rx=0, ry=0, rz=0):
        return Frames(f.turn(rx, ry, rz) for f in self)

    def move(self, x=0, y=0, z=0):
        return Frames(f.move(x, y, z) for f in self)

    def __getitem__(self, i):
        got = super().__getitem__(i)
        return Frames(got) if isinstance(i, slice) else got

    def __add__(self, other):
        joined = list(self) + list(other)
        return Frames(joined) if all(isinstance(f, Frame) for f in joined) else joined

    def __radd__(self, other):
        joined = list(other) + list(self)
        return Frames(joined) if all(isinstance(f, Frame) for f in joined) else joined

    def __mul__(self, n):
        return Frames(list(self) * n)

    __rmul__ = __mul__


def _one(on, verb):
    """`on` as one frame (the origin's if None)."""
    if on is None:
        return Frame()
    if not hasattr(on, "matrix") or isinstance(on, (list, tuple)):
        raise TypeError(f"{verb}: on is one frame, not {type(on).__name__}: a grid or a ring is laid on one plane")
    return on


def _frame_of(matrix):
    return Frame(matrix.to_translation(), matrix.col[2].xyz, matrix.col[1].xyz)


# ---------------------------------------------------------------- the frame of a face

_DIRECTION = re.compile(r"^([+\-<>]?)([xyzXYZ])$")


def face(ob, which):
    """Frame of a named face, or furthest along "+Z".

    The frame of the face of `ob` named `which` ("top", "boss.top", "cut") — the
    largest flat group of faces with that name — or, given a direction ("+Z", "-X"),
    of the face furthest that way. Its origin is the parent's origin projected onto
    the face's plane, as CadQuery places a workplane."""
    me = ob.data
    polys = list(me.polygons)
    d = _DIRECTION.match(which)
    if d:
        chosen = _facing(ob, which)
        if not chosen:
            raise ValueError(f"{_core.given_name(ob)} has no face facing {which}")
    else:
        att = me.attributes.get(_core.FACE_TAG + which)
        if att is None:
            names = sorted(a.name[len(_core.FACE_TAG):] for a in me.attributes if a.name.startswith(_core.FACE_TAG))
            raise ValueError(_core.unknown(_core.given_name(ob), "face", [which], names))
        tagged = [p for p, v in zip(polys, att.data) if v.value]
        groups = {}
        for p in tagged:
            n = p.normal
            key = (round(n.x, 4) + 0.0, round(n.y, 4) + 0.0, round(n.z, 4) + 0.0,
                   round(_core.in_mm(n.dot(p.center)), 4) + 0.0)
            groups.setdefault(key, []).append(p)
        if not groups:
            raise ValueError(f"{_core.given_name(ob)}: no face carries {which!r} any more")
        chosen = max(groups.values(), key=lambda g: (sum(p.area for p in g), -min(p.index for p in g)))
        total = sum(p.area for p in tagged)
        if len(groups) > 6 and sum(p.area for p in chosen) < 0.5 * total:
            raise ValueError(f"{_core.given_name(ob)}: the faces named {which!r} are not flat")
    normal, point = _plane(me, chosen, ob.matrix_basis)
    if normal is None:
        raise ValueError(f"{_core.given_name(ob)}: the face {which!r} has no area")
    z = Frame((0, 0, 0), normal).z      # its axes cleaned of dust, as the frame will have them
    offset = sum(c * p for c, p in zip(z, point))
    return Frame([c * offset for c in z], normal)._read_on(ob, ob.matrix_basis)


def _facing(ob, which):
    """The faces of `ob` looking the direction `which` ("+Z", "-X", in the space `ob` is in) within
    1°, and furthest that way: what `face` and `shell(open=)` read a direction as. [] if none."""
    d = _DIRECTION.match(which)
    axis = Vector({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[d.group(2).lower()])
    if d.group(1) in ("-", "<"):
        axis = -axis
    rot = ob.matrix_basis.to_3x3()
    local = (rot.inverted() @ axis).normalized()
    chosen = [p for p in ob.data.polygons if p.normal.dot(local) > math.cos(math.radians(1))]
    if not chosen:
        return []
    far = max(p.center.dot(local) for p in chosen)
    return [p for p in chosen if p.center.dot(local) > far - 1e-6 * max(_core.mm(1.0), abs(far))]


def _plane(me, polys, m):
    """The plane of flat faces, placed by `m`: a normal and a point on it, in double
    precision from the mesh's points; the normal is None if the faces have no area.

    Blender's own normal and centre of an n-gon are single-precision sums: the top of a
    12 mm disc came out at z = 12.0000095, and a part stood on it floated off it.
    The normal is Newell's over every face (so weighted by area), the point the mean of
    their corners."""
    pts, n = [], [0.0, 0.0, 0.0]
    for poly in polys:
        corners = [tuple(me.vertices[i].co) for i in poly.vertices]
        for (x0, y0, z0), (x1, y1, z1) in zip(corners, corners[1:] + corners[:1]):
            n[0] += (y0 - y1) * (z0 + z1)
            n[1] += (z0 - z1) * (x0 + x1)
            n[2] += (x0 - x1) * (y0 + y1)
        pts += corners
    centre = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
    a = [[m[row][col] for col in range(3)] for row in range(3)]
    # a normal goes through the cofactor matrix: the inverse transpose, up to the determinant
    cof = [[a[(r + 1) % 3][(c + 1) % 3] * a[(r + 2) % 3][(c + 2) % 3]
            - a[(r + 1) % 3][(c + 2) % 3] * a[(r + 2) % 3][(c + 1) % 3] for c in range(3)] for r in range(3)]
    det = sum(a[0][c] * cof[0][c] for c in range(3))
    normal = [math.copysign(1.0, det) * sum(cof[r][c] * n[c] for c in range(3)) for r in range(3)]
    size = math.sqrt(sum(c * c for c in normal))
    point = [sum(a[r][c] * centre[c] for c in range(3)) + m[r][3] for r in range(3)]
    return ([c / size for c in normal] if size > 0 else None), point


#: Points of an edge path closer than this (mm) are one: a boolean leaves slivers along its
#: seams, and a step of no length has no direction to sweep along.
EDGE_WELD = 0.02


def edge(ob, which):
    """Path along the edges named which, in the space ob is in: the longest run (as long: the one reaching
    furthest -X, then -Y, -Z), closed if it closes.

    The edges of `ob` named `which` ("top", "cut", a cut tool's name, "plate.top") as a
    path, in the space `ob` is in, as `ob.face` reads a face: the longest run of them, a
    closed path if it closes — for `wire`, `sweep` and `along`: a rolled rim along the
    line a trim left, `s3d.wire("lip", tub.edge("above_rim"), d=56)`. A closed run turns
    anticlockwise seen from +Z (from +Y, from +X if it stands that way), from where it
    crosses its own +X (+Z, +Y); an open one runs toward +X (then +Y, +Z). Edges that fork
    raise: a path follows a line, not a junction."""
    me = ob.data
    att = me.attributes.get(_core.EDGE_TAG + which) if isinstance(which, str) else None
    if att is None or att.domain != "EDGE":
        names = sorted(a.name[len(_core.EDGE_TAG):] for a in me.attributes if a.name.startswith(_core.EDGE_TAG))
        raise ValueError(_core.unknown(_core.given_name(ob), "edge", [str(which)], names))
    links = {}
    for e, flag in zip(me.edges, att.data):
        if flag.value:
            a, b = e.vertices
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)
    if not links:
        raise ValueError(f"{_core.given_name(ob)}: no edge carries {which!r} any more")
    forks = sorted(v for v, n in links.items() if len(n) > 2)
    if forks:
        raise ValueError(f"{_core.given_name(ob)}: the edges named {which!r} fork at {len(forks)} point(s) — a path "
                         f"follows a line, not a junction: name one loop or one run (a single cut's)")
    m = ob.matrix_basis
    co = {v: m @ me.vertices[v].co for v in links}
    runs, seen = [], set()
    for start in sorted(links, key=lambda v: (len(links[v]) != 1, v)):   # open runs from an end
        if start in seen:
            continue
        run, here = [start], start
        seen.add(start)
        while True:
            nxt = [v for v in links[here] if v not in seen]
            if not nxt:
                break
            here = min(nxt)
            run.append(here)
            seen.add(here)
        closed = len(links[start]) == 2
        pts = [co[run[0]]]
        for v in run[1:]:
            if (co[v] - pts[-1]).length > _core.mm(EDGE_WELD):
                pts.append(co[v])
        if closed and len(pts) > 1 and (pts[-1] - pts[0]).length <= _core.mm(EDGE_WELD):
            pts.pop()
        length = sum((b - a).length for a, b in zip(pts, pts[1:] + (pts[:1] if closed else [])))
        runs.append((length, closed, pts))
    # the longest; of runs as long, the one reaching furthest -X (then -Y, -Z): one answer, whatever the order
    length, closed, pts = max(runs, key=lambda r: (round(_core.in_mm(r[0]), 6),
                                                   tuple(-round(_core.in_mm(c), 6) for c in min(tuple(q) for q in r[2]))))
    if len(pts) < (3 if closed else 2):
        raise ValueError(f"{_core.given_name(ob)}: the edges named {which!r} are too short to make a path")
    return Path([tuple(p) for p in (_loop_order(pts) if closed else _run_order(pts))], closed=closed)


def _loop_order(pts):
    """A closed run anticlockwise about the axis its area faces most (Z, then Y, then X), from
    the point nearest the ray from its centre along that view's first axis."""
    n = Vector()
    for a, b in zip(pts, pts[1:] + pts[:1]):
        n += a.cross(b)
    axis = max((2, 1, 0), key=lambda k: abs(n[k]) + 1e-9 * k)
    if n[axis] < 0:
        pts = pts[::-1]
    u, v = {2: (0, 1), 1: (2, 0), 0: (1, 2)}[axis]
    c = sum(pts, Vector()) / len(pts)
    first = min(range(len(pts)), key=lambda i: (round(math.atan2(pts[i][v] - c[v], pts[i][u] - c[u])
                                                      % (2 * math.pi), 9), i))
    return pts[first:] + pts[:first]


def _run_order(pts):
    """An open run toward +X, then +Y, then +Z."""
    key = [tuple(round(_core.in_mm(c), 6) for c in p) for p in (pts[0], pts[-1])]
    return pts[::-1] if key[1] < key[0] else pts


_SIDES = re.compile(r"[+\-<>]?[xyzXYZ]")


def _sides(which, what):
    """`which` — "+X", "+X-Y", "-Y+X+Z" — as [(axis, ±1)]: one to three different axes,
    a sign on each (a lone axis may go without: "+")."""
    terms = _SIDES.findall(which) if isinstance(which, str) else []
    if (not terms or "".join(terms) != which or len(terms) > 3
            or (len(terms) > 1 and any(t[0] not in "+-<>" for t in terms))
            or len({t[-1].lower() for t in terms}) < len(terms)):
        raise ValueError(f"{what}: {which!r} names no side of a box: a side \"+X\", an edge \"+X-Y\" or a corner "
                         f"\"+X-Y+Z\" — a sign and an axis for each, every axis once")
    return [("xyz".index(t[-1].lower()), -1 if t[0] in "-<" else 1) for t in terms]


def _placed_points(ob):
    """Every mesh point under `ob` (itself included), in the space `ob` is in: the
    placements composed and snapped as `_core.world` does, the points moved in double
    precision."""
    out = []

    def walk(o, m):
        if o.type == "MESH":
            rows = [[m[r][c] for c in range(4)] for r in range(3)]
            co = [0.0] * (3 * len(o.data.vertices))
            o.data.vertices.foreach_get("co", co)
            for i in range(0, len(co), 3):
                x, y, z = co[i], co[i + 1], co[i + 2]
                out.append(tuple(r[0] * x + r[1] * y + r[2] * z + r[3] for r in rows))
        for c in o.children:
            walk(c, _core.snapped(m @ c.matrix_parent_inverse @ c.matrix_basis))
    walk(ob, _core.snapped(Matrix(ob.matrix_basis)))
    return out


def bbox(ob, which):
    """Frame on its bounding box: a side "+X", an edge "+X-Y", a corner "+X-Y+Z".

    A frame on the box that bounds `ob` — a part, a group or an assembly, everything
    under it — in the space it is in, its sides along that space's axes: `"+X"` is the
    centre of the side looking +X, `"+X-Y"` the middle of the edge it shares with the
    side looking -Y, `"+X-Y+Z"` a corner. The frame has the axes of a face looking the
    first way named — its normal that way — and is read on `ob`, as `ob.face` is."""
    from .assembly import Handle
    root = ob.root if isinstance(ob, Handle) else ob
    if not isinstance(root, bpy.types.Object):
        raise TypeError(f"bbox: a part, a group or an assembly is expected, not {type(ob).__name__}")
    sides = _sides(which, "bbox")
    pts = _placed_points(root)
    if not pts:
        raise ValueError(f"bbox: {_core.given_name(root)!r} holds no solid: it has no box")
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    origin = [(lo[k] + hi[k]) / 2 for k in range(3)]
    for axis, sign in sides:
        origin[axis] = hi[axis] if sign > 0 else lo[axis]
    normal = [0.0, 0.0, 0.0]
    normal[sides[0][0]] = float(sides[0][1])
    return Frame(origin, normal)._read_on(root, root.matrix_basis)


def local(ob, x=0, y=0, z=0, rx=0, ry=0, rz=0):
    """Frame at (x, y, z) of its own space, turned (rx, ry, rz) degrees.

    A frame on `ob` — a part, a group or an assembly — at (x, y, z) of `ob`'s own
    space, the one it was built in, turned (rx, ry, rz) degrees as `at` turns: a
    source's joint origin, its xyz and rpy, as written. Like `ob.face` and `bbox`,
    it is in the space `ob` is in and read on `ob`: `at(part, on=…)` puts a part there,
    and a joint on it takes its pivot from it, wherever `ob` has gone since."""
    from .assembly import Handle
    root = ob.root if isinstance(ob, Handle) else ob
    if not isinstance(root, bpy.types.Object):
        raise TypeError(f"local: a part, a group or an assembly is expected, not {type(ob).__name__}: the frame is "
                        f"read on it, at a point of its own space")
    values = dict(zip(KEYS, (x, y, z, rx, ry, rz)))
    for name, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"local: {name} is a number, not {v!r}")
    basis = _core.snapped(Matrix(root.matrix_basis))
    return _frame_of(_core.snapped(basis @ _turned(values)))._read_on(root, basis)


def _frame_only(method):
    def refuse(ob, *args, **kwargs):
        raise TypeError(f"{_core.given_name(ob)!r} is a solid: it moves with s3d.transform(ob, x=…, rz=…), or "
                        f"s3d.at(ob, …) places it; .{method}() is a frame's")
    return refuse


bpy.types.Object.face = face
bpy.types.Object.edge = edge
for _method in ("move", "turn", "offset"):
    setattr(bpy.types.Object, _method, _frame_only(_method))
# bpy 4.5 never exits while a Python attribute sits on an RNA type: take them off first.
atexit.register(lambda: [delattr(bpy.types.Object, a) for a in ("face", "edge", "move", "turn", "offset")
                         if hasattr(bpy.types.Object, a)])


# ---------------------------------------------------------------- placing

#: Anchor words and the bounding-box side they pick on each axis.
_ANCHOR_WORDS = {"left": (0, -1), "right": (0, 1), "front": (1, -1), "back": (1, 1),
                 "bottom": (2, -1), "top": (2, 1), "center": None, "centre": None}


def _anchor_point(lo, hi, anchor):
    """The point of the box (lo, hi) an anchor names: "bottom", "top-front", "bottom-left-back"…
    or the directions `bbox` takes: "-Z", "+Y-Z", "+X-Y+Z"."""
    p = (lo + hi) / 2
    if anchor in (None, "origin"):
        return Vector((0, 0, 0))
    if isinstance(anchor, str) and re.fullmatch(r"([+\-<>][xyzXYZ])+|[xyzXYZ]", anchor.strip()):
        for axis, side in _sides(anchor.strip(), f"anchor {anchor!r}"):
            p[axis] = hi[axis] if side > 0 else lo[axis]
        return p
    for word in re.split(r"[-_ ]+", anchor.strip().lower()):
        if word not in _ANCHOR_WORDS:
            raise ValueError(f"anchor {anchor!r}: {word!r} is none of {sorted(_ANCHOR_WORDS)}, and the anchor is "
                             f"no directions either (\"-Z\", \"+Y-Z\", \"+X-Y+Z\")")
        if _ANCHOR_WORDS[word]:
            axis, side = _ANCHOR_WORDS[word]
            p[axis] = hi[axis] if side > 0 else lo[axis]
    return p


def _bounds(ob):
    """(lo, hi) of everything under `ob`, in `ob`'s own frame."""
    pts = []

    def walk(o, m):
        if o.type == "MESH":
            pts.extend(m @ v.co for v in o.data.vertices)
        for c in o.children:
            walk(c, m @ c.matrix_parent_inverse @ c.matrix_basis)
    walk(ob, Matrix.Identity(4))
    if not pts:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def _mirror(mirror):
    s = [1.0, 1.0, 1.0]
    for c in (mirror or "").lower():
        if c not in "xyz":
            raise ValueError(f"transform: mirror is among 'x', 'y', 'z' — the coordinates that change sign — not "
                             f"{mirror!r}")
        s["xyz".index(c)] = -1.0
    return Matrix.Diagonal((*s, 1.0))


#: Custom property: the frame, the values and the anchor point `at` last placed an object with.
PLACE = "s3d~place"
KEYS = ("x", "y", "z", "rx", "ry", "rz")


def frames_of(on, verb, param="on"):
    """`on` as a list of frames, for `verb`'s parameter `param`; anything else raises."""
    if not isinstance(on, (list, tuple)) or not on or not all(isinstance(f, Frame) for f in on):
        raise TypeError(f"{verb}: {param} is a frame or a list of frames, as grid(), ring() and along() give")
    return list(on)


def _elements(ob):
    """The things a list stands for, nested lists opened — objects, handles, profiles,
    paths — or None if `ob` is not such a list (a list of points is a path)."""
    from .assembly import Handle
    if not isinstance(ob, (list, tuple)) or not ob:
        return None
    out = []
    for item in ob:
        if isinstance(item, (list, tuple)):
            inner = _elements(item)
            if inner is None:
                return None
            out += inner
        elif isinstance(item, (bpy.types.Object, Handle, Profile, Placed, Path)):
            out.append(item)
        else:
            return None
    return out


def _turned(values):
    v = values
    return Matrix.Translation((v["x"], v["y"], v["z"])) @ Euler(
        (math.radians(v["rx"]), math.radians(v["ry"]), math.radians(v["rz"]))).to_matrix().to_4x4()


def at(ob, x=None, y=None, z=None, rx=None, ry=None, rz=None, on=None, anchor=None):
    """Place it: set what is given, keep the rest; anchor: "-Z", "+Y-Z"….

    Put `ob` at (x, y, z), turned (rx, ry, rz) degrees, in the frame `on` — or in the
    frame it is in already (its parent's, at first). Only the values given change: the
    others keep what `at` set before (0 at first), and `on` changes the frame, keeping
    them. `anchor` names the point of its bounding box that goes there: directions, as
    `bbox` takes them ("-Z", "+Y-Z", "+X-Y+Z"), or words ("bottom", "top-front",
    "bottom-left-back": top bottom left right front back are +Z -Z -X +X -Y +Y); its own
    origin by default. Returns `ob`: an
    object, a handle, a profile (placed — a loft's section) or a path (moved); each of
    a list. With a list of frames for `on`, `ob` is used up and one copy goes on each
    frame: the list of copies is returned (of placed profiles, for a profile). A mirror
    image is `transform`'s."""
    from .assembly import Handle
    given = {k: v for k, v in zip(KEYS, (x, y, z, rx, ry, rz)) if v is not None}
    if isinstance(on, (list, tuple)):
        return _on_frames(ob, frames_of(on, "at"), given, anchor)
    if on is not None and not hasattr(on, "matrix"):
        raise TypeError(f"at: on is a frame or a list of frames, as grid(), ring() and along() give, not {on!r}")
    items = _elements(ob)
    if items is not None:
        if anchor is not None and any(isinstance(i, (bpy.types.Object, Handle)) for i in items):
            raise ValueError("at: anchor on a list of objects: each has its own box — place them one by one")
        return [at(i, x, y, z, rx, ry, rz, on, anchor) for i in items]
    frame = on.matrix if on is not None else None
    if isinstance(ob, Handle):
        at(ob.root, x, y, z, rx, ry, rz, on, anchor)
        return ob
    if isinstance(ob, (Profile, Placed, Path, list, tuple)):
        return _replaced(ob, frame, given)
    return _posed(ob, frame, given, anchor)


def _posed(ob, frame, given, anchor):
    """`ob` placed by what it was placed with, `given` and the rest replacing it. The
    anchor's point is the one found when the anchor was given: moving the part does not
    re-anchor it on a box that grew since — giving `anchor` again does."""
    if not (given or frame is not None or anchor is not None):
        return ob                                   # `at(ob)`: nothing to change
    from .joints import refuse_at
    refuse_at(ob)
    kept_frame, values, kept_anchor, point = _recalled(ob)
    if frame is not None:
        kept_frame = frame
    values.update(given)
    if anchor is not None:
        kept_anchor, point = anchor, _anchor_point(*_bounds(ob), anchor)
    ob.matrix_basis = _pose(kept_frame, values, point)
    _remember(ob, kept_frame, values, kept_anchor, point)
    return ob


def _pose(frame, values, point):
    m = _turned(values)
    if frame is not None:
        m = frame @ m
    return m @ Matrix.Translation(-point)


def _replaced(ob, frame, given):
    """A profile or a path placed again: the frame and values it was placed with, `given`
    replacing them, over what it was placed from. (Not `_pose`: a profile has no anchor,
    and a single call must give the matrix it always gave.)"""
    kept_frame, values, base = _recalled_value(ob)
    if frame is not None:
        kept_frame = frame
    values.update(given)
    full = _turned(values)
    if kept_frame is not None:
        full = kept_frame @ full
    if isinstance(ob, (Profile, Placed)):
        out = Placed(ob.profile if isinstance(ob, Placed) else ob, full @ base)
    else:
        out = Path([tuple(full @ Vector(p)) for p in base.points], closed=base.closed, sharp=base.sharp)
    out._s3d_place = (kept_frame, values, base)
    return out


def _recalled_value(ob):
    """(frame, values, base) a profile or a path was placed with; never placed: no frame,
    values 0, and itself as the base."""
    kept = getattr(ob, "_s3d_place", None)
    if kept is not None:
        frame, values, base = kept
        return frame, dict(values), base
    base = ob.matrix if isinstance(ob, Placed) else (Matrix.Identity(4) if isinstance(ob, Profile) else _as_path(ob))
    return None, dict.fromkeys(KEYS, 0.0), base


def transform(ob, x=0, y=0, z=0, rx=0, ry=0, rz=0, mirror=None):
    """Move it by, as a parent would; its placement, not its mesh.

    Add to where `ob` is: moved by (x, y, z), turned (rx, ry, rz) degrees about its
    parent's origin, `mirror` ("x", "yz"…) its image across those planes of its parent —
    what `at` on a group around it would do. The placement changes, not the mesh; a
    jointed part moves with its pivot, and its mirror image moves as the image of the
    motion. Returns
    `ob`: an object, a handle (its root moves), each of a list, or a new profile or path.
    It moves the frame `ob` is in: a later `at(ob, x=…)` places it in the moved frame,
    `at(ob, on=…)` in a new one."""
    if isinstance(ob, Frame) or isinstance(ob, (list, tuple)) and ob and all(isinstance(f, Frame) for f in ob):
        raise TypeError("transform: a frame moves with its own .move(), .turn() and .offset(), and so does a list "
                        "of frames")
    items = _elements(ob)
    if items is not None:
        _not_nested(items, "transform")
    return _transform(ob, _turned(dict(zip(KEYS, (x, y, z, rx, ry, rz)))) @ _mirror(mirror))


def _not_nested(items, verb):
    """Refuse a list holding a group and one of its own parts: the part would go twice."""
    from .assembly import Handle
    roots = {(i.root if isinstance(i, Handle) else i).as_pointer() for i in items
             if isinstance(i, (bpy.types.Object, Handle))}
    for i in items:
        o = i.root if isinstance(i, Handle) else i
        parent = getattr(o, "parent", None)
        while parent is not None:
            if parent.as_pointer() in roots:
                raise ValueError(f"{verb}: {_core.given_name(o)!r} is inside {_core.given_name(parent)!r}, which "
                                 f"this call moves whole: give one of them")
            parent = parent.parent


def _transform(ob, m):
    from .assembly import Handle
    items = _elements(ob)
    if items is not None:
        return [_transform(i, m) for i in items]
    if isinstance(ob, Handle):
        _transform(ob.root, m)
        return ob
    if isinstance(ob, (Profile, Placed, Path, list, tuple)):
        frame, values, base = _recalled_value(ob)
        if isinstance(ob, (Profile, Placed)):
            prof, inner = (ob.profile, ob.matrix) if isinstance(ob, Placed) else (ob, Matrix.Identity(4))
            out = Placed(prof, m @ inner)
        else:
            path = _as_path(ob)
            out = Path([tuple(m @ Vector(p)) for p in path.points], closed=path.closed, sharp=path.sharp)
        out._s3d_place = (m @ frame if frame is not None else Matrix(m), values, base)   # the frame moves
        return out
    kept = _recalled(ob) if ob.get(PLACE) is not None else None
    ob.matrix_basis = m @ ob.matrix_basis
    from .joints import moved
    moved(ob, m)                         # a jointed part moves with its pivot, at rest as posed
    if kept is not None:                 # the frame `at` placed it in moves with it
        frame, values, anchor, point = kept
        _remember(ob, m @ frame if frame is not None else Matrix(m), values, anchor, point)
    return ob


def _flat16(m):
    return [m[r][c] for r in range(4) for c in range(4)]


def _matrix16(f):
    return Matrix([list(f[4 * r:4 * r + 4]) for r in range(4)])


def _remember(ob, frame, values, anchor, point):
    """Keep on `ob` the frame (None: its parent's), the values, the anchor and the point
    of its box the anchor named, that it was placed with — and the placement Blender
    holds now, to tell later whether something else moved it."""
    ob[PLACE] = {**{k: float(v) for k, v in values.items()}, "anchor": anchor or "",
                 "point": [float(c) for c in point], "frame": [] if frame is None else _flat16(frame),
                 "basis": _flat16(ob.matrix_basis)}


def _recalled(ob):
    """(frame, values, anchor, anchor point) `at` last placed `ob` with. Never
    placed with `at` — or moved since by something else than sand3d — the placement it has
    is its frame, its values 0 (no frame at all for an object at its parent's origin)."""
    kept = ob.get(PLACE)
    if kept is not None:
        kept = kept.to_dict()
        frame = _matrix16(kept["frame"]) if len(kept["frame"]) == 16 else None
        values, anchor, point = {k: kept[k] for k in KEYS}, kept["anchor"] or None, Vector(kept["point"])
        now = Matrix(ob.matrix_basis)
        if _flat16(now) != list(kept.get("basis", [])) or kept.get("mirror"):
            # (a placement 0.x stored with a mirror is taken where it is, mirror and all)
            # moved by something else since (raw bpy): the move goes to the frame, the values stay
            frame = now @ _pose(None, values, point).inverted()
        return frame, values, anchor, point
    frame = Matrix(ob.matrix_basis)
    return (None if frame == Matrix.Identity(4) else frame), dict.fromkeys(KEYS, 0.0), None, Vector((0, 0, 0))


def _on_frames(ob, frames, given, anchor):
    """`at(ob, on=[frames])`: `ob` used up, a copy on each frame — placed as `at` with one
    frame would place it."""
    from . import _guard
    from .assembly import Handle
    from .edit import _children, _copy, _delete_tree
    items = _elements(ob)
    if items is not None:
        _not_nested(items, "at")
        return [c for i in items for c in _on_frames(i, frames, given, anchor)]
    if not isinstance(ob, (bpy.types.Object, Handle)):
        return [_replaced(ob, f.matrix, given) for f in frames]
    _anchor_point(Vector(), Vector(), anchor)          # a wrong word raises before anything is made
    root = ob.root if isinstance(ob, Handle) else ob
    from .joints import refuse_at
    refuse_at(root)
    children = _children() if root.children else {}
    out = []
    try:
        for f in frames:
            twin = _copy(ob, children)
            _posed(twin.root if isinstance(twin, Handle) else twin, f.matrix, given, anchor)
            out.append(twin)
    except Exception:
        for twin in out:
            _delete_tree(twin.root if isinstance(twin, Handle) else twin)
        raise
    tree = [root]
    for o in tree:
        tree += children.get(o, [])
    _guard.using_up(tree, root)
    _delete_tree(root)
    return out


# ---------------------------------------------------------------- distributors

def grid(nx, ny=1, px=0, py=0, stagger=False, on=None):
    """On a grid.

    `nx` × `ny` frames `px` and `py` apart, centred on `on` (the origin by default);
    `stagger` shifts every other row by half a pitch."""
    base = _one(on, "grid")
    out = []
    for j in range(ny):
        shift = px / 2 if stagger and j % 2 else 0
        for i in range(nx):
            centre = px / 4 if stagger and ny > 1 else 0
            out.append(base.move((i - (nx - 1) / 2) * px + shift - centre, (j - (ny - 1) / 2) * py))
    return Frames(out)


def ring(n, r, start=0, angle=360, on=None):
    """On a circle.

    `n` frames on a circle of radius `r` round `on`'s normal, from `start` degrees
    (0 on its X): spread over `angle` degrees, evenly round a full turn. Each frame's
    X points outward, so a part placed on it turns with the ring."""
    base = _one(on, "ring")
    step = angle / n if abs(angle - 360) < 1e-9 else (angle / (n - 1) if n > 1 else 0)
    out = []
    for k in range(n):
        a = math.radians(start + step * k)
        radial = base.x * math.cos(a) + base.y * math.sin(a)
        out.append(base._kept(Frame(base.origin + radial * r, base.z, base.z.cross(radial))))
    return Frames(out)


def along(path, pitch=None, count=None):
    """Along a path.

    Frames along `path`, facing the way it runs: every `pitch` from its start, or
    `count` of them evenly from end to end (round a closed path)."""
    path = _as_path(path)
    pts = [Vector(p) for p in path.points]
    if path.closed:
        pts = pts + [pts[0]]
    lengths = [0.0]
    for a, b in zip(pts, pts[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    total = lengths[-1]
    if count is not None:
        spots = [total * k / (count if path.closed else max(count - 1, 1)) for k in range(count)]
    elif pitch:
        spots = [pitch * k for k in range(int(total / pitch + 1e-6) + 1)]   # lengths are single precision
    else:
        raise ValueError("along: give a pitch or a count")
    out, j = [], 0
    for s in spots:
        while j < len(pts) - 2 and lengths[j + 1] < s:
            j += 1
        seg = lengths[j + 1] - lengths[j]
        f = (s - lengths[j]) / seg if seg > 0 else 0.0
        tangent = (pts[j + 1] - pts[j]).normalized()
        out.append(Frame(pts[j].lerp(pts[j + 1], f), tangent))
    return Frames(out)


def snap(targets, points, direction=None):
    """On a surface.

    Frames on the surface of `targets` (one object or several), one per point or
    frame of `points`: at the nearest spot, or where a line from it running along
    `direction` first meets the surface; each frame's normal points out of it. Points
    a line misses are dropped."""
    size = _core.in_mm(1.0)   # Blender's BVH pads its boxes in its own units, which breaks ties: search in mm
    trees = [_core.mm_tree(ob, ob.matrix_basis) for ob in (targets if isinstance(targets, (list, tuple)) else [targets])]
    out = []
    for p in points:
        p = Vector(p.origin if isinstance(p, Frame) else p) * size
        if direction is None:
            found = [t.find_nearest(p) for t in trees]
            loc, normal, _, _ = min((f for f in found if f[0] is not None), key=lambda f: f[3])
            if normal.dot(p - loc) < 0:
                normal = -normal
        else:
            ray = Vector(direction).normalized()
            found = [f for f in (t.ray_cast(p, ray) for t in trees) if f[0] is not None]
            if not found:
                continue
            loc, normal, _, _ = min(found, key=lambda f: f[3])
            if normal.dot(ray) > 0:
                normal = -normal
        out.append(Frame(loc / size, normal))
    return Frames(out)
