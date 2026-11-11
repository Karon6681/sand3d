"""Deterministic array noise and conforming tetrahedral level surfaces."""

import math
import itertools
import bpy
import numpy as np
from . import _core
from .edit import NOISES


def _mix(h, seed):
    # Own the buffer: callers share the coordinate products across neighbours.
    h = np.asarray(h ^ np.uint32(seed & 0xFFFFFFFF), dtype=np.uint32)
    with np.errstate(over="ignore"):
        h ^= h >> np.uint32(16)
        h *= np.uint32(2246822519)
        h ^= h >> np.uint32(13)
        h *= np.uint32(3266489917)
        h ^= h >> np.uint32(16)
    return h


def noise(x, y, z, seed=0, kind="perlin"):
    """Float64 noise in [-1, 1]; scalars or broadcastable arrays.

    Seeded gradient / nearest-cell noise, stable across runs and bpy versions.
    Coordinates are dimensionless. This is independent of displace's noise.
    """
    if kind not in NOISES:
        raise ValueError(f"noise: kind must be one of {NOISES}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("noise: seed must be an integer")
    q = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in (x, y, z)])
    if any(not np.all(np.isfinite(a)) or np.any(np.abs(a) > 2**50) for a in q):
        raise ValueError("noise: coordinates must be finite and within 2**50")
    base = [np.floor(a).astype(np.int64) for a in q]
    frac = [a - b for a, b in zip(q, base)]
    steps = (-1, 0, 1) if kind == "cells" else (0, 1)
    with np.errstate(over="ignore"):
        products = [
            {d: (b.astype(np.uint32) + np.uint32(d & 0xFFFFFFFF)) * np.uint32(k)
             for d in steps}
            for b, k in zip(base, (1597334677, 3812015801, 2798796415))
        ]
    if kind == "cells":
        distance = np.full(q[0].shape, np.inf)
        for delta in itertools.product((-1, 0, 1), repeat=3):
            hashed = products[0][delta[0]] ^ products[1][delta[1]] ^ products[2][delta[2]]
            dist = np.zeros(q[0].shape)
            for c in range(3):
                offset = (
                    _mix(hashed, seed + 1013 * (c + 1)).astype(np.float64) / 4294967296
                )
                dist += (delta[c] + offset - frac[c]) ** 2
            np.minimum(distance, dist, out=distance)
        result = 1 - 2 * np.minimum(1, np.sqrt(distance) / math.sqrt(3))
    else:
        fade = [f * f * f * (f * (f * 6 - 15) + 10) for f in frac]
        result = np.zeros(q[0].shape)
        for corner in itertools.product((0, 1), repeat=3):
            h = _mix(products[0][corner[0]] ^ products[1][corner[1]] ^ products[2][corner[2]], seed)
            # Twelve edge directions of the cube, constructed without a table.
            axis = h % 3
            dot = np.zeros(q[0].shape)
            for c in range(3):
                term = np.asarray(frac[c] - corner[c])
                np.negative(term, out=term, where=((h >> (c + 3)) & 1) == 0)
                np.copyto(term, 0.0, where=axis == c)
                dot += term
            weight = np.ones(q[0].shape)
            for c in range(3):
                weight *= fade[c] if corner[c] else 1 - fade[c]
            result += weight * dot
        result = np.clip(result, -1, 1)
        if kind == "ridged":
            result = 1 - 2 * np.abs(result)
        if kind == "turbulence":
            result = 2 * np.abs(result) - 1
    return float(result) if result.ndim == 0 else result


DENSE_SAMPLES = 1_000_000
MAX_COARSE_SAMPLES = 1_000_000
MAX_SAMPLES = 8_000_000
MAX_TRIANGLES = 4_000_000
_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def _tetrahedron_cases():
    """Derive oriented tetrahedron cases from cube geometry."""
    cube = np.array(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=np.float64,
    )
    cases = []
    for tet in _TETS:
        xyz = cube[list(tet)]
        table = {}
        for code in range(1, 15):
            inside = [c for c in range(4) if code & (1 << c)]
            outside = [c for c in range(4) if not code & (1 << c)]
            if len(inside) == 1 or len(outside) == 1:
                one, many = (inside, outside) if len(inside) == 1 else (outside, inside)
                patterns = [[(one[0], c) for c in many]]
            else:
                a, b = inside
                c, d = outside
                patterns = [[(a, c), (a, d), (b, d)], [(a, c), (b, d), (b, c)]]
            oriented = []
            for pattern in patterns:
                pattern = np.array(pattern)
                mid = xyz[pattern].mean(axis=1)
                normal = np.cross(mid[1] - mid[0], mid[2] - mid[0])
                if normal.dot(xyz[outside[0]] - xyz[inside[0]]) < 0:
                    pattern = pattern[[0, 2, 1]]
                oriented.append(pattern)
            table[code] = oriented
        cases.append(table)
    return cases


_CASES = _tetrahedron_cases()


def field(name, size, density, *, cell, level=0, min_component_cells=3):
    """Closed level surface; see docs/api.md for sampling and array contracts.

    Fixed facets(n) uses a cell limit max(cell, π max(size) / n).
    """
    try:
        size = tuple(float(v) for v in size)
        cell, level = float(cell), float(level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("field: size, cell and level must be finite numbers") from exc
    if (
        len(size) != 3
        or not all(math.isfinite(v) and v > 0 for v in (*size, cell))
        or not math.isfinite(level)
    ):
        raise ValueError(
            "field: size has three positive finite lengths; cell is positive and level finite"
        )
    if not callable(density):
        raise TypeError("field: density must be callable")
    cell = max(cell, _core.surface_edge(size))
    ratios = [v / cell for v in size]
    if any(not math.isfinite(v) or v > MAX_SAMPLES for v in ratios):
        raise ValueError("field: grid axis exceeds sample budget; increase cell")
    ns = [max(2, math.ceil(v)) + 1 for v in ratios]
    nx, ny, nz = ns
    if math.prod((n + 2) // 4 + 1 for n in ns) > MAX_COARSE_SAMPLES:
        raise ValueError("field: coarse grid exceeds 1,000,000 probes; increase cell")
    if not math.isfinite(min_component_cells) or min_component_cells < 0:
        raise ValueError("field: min_component_cells must be finite and nonnegative")
    axes = [
        np.linspace(-v / 2 if c < 2 else 0, v / 2 if c < 2 else v, ns[c])
        for c, v in enumerate(size)
    ]
    vectorized = None

    def evaluate(points):
        nonlocal vectorized
        if vectorized is None:
            try:
                probe = density(*points[: min(8, len(points))].T)
                vectorized = isinstance(probe, np.ndarray) and probe.shape == (
                    min(8, len(points)),
                )
            except (TypeError, ValueError):
                vectorized = False
        try:
            if vectorized:
                out = np.asarray(density(*points.T), dtype=np.float64)
                if out.shape != (len(points),):
                    raise ValueError("density array shape changed")
            else:
                out = np.fromiter(
                    (float(density(*p)) for p in points), np.float64, len(points)
                )
            out = out - level
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("field: invalid or non-finite sample") from exc
        if not np.all(np.isfinite(out)):
            raise ValueError("field: non-finite sample")
        return out

    plane = nx * ny

    def points(ids):
        return np.column_stack(
            (axes[0][ids % nx], axes[1][ids // nx % ny], axes[2][ids // plane])
        )

    def sample(ids):
        v = evaluate(points(ids))
        boundary = (
            (ids % nx == 0)
            | (ids % nx == nx - 1)
            | (ids // nx % ny == 0)
            | (ids // nx % ny == ny - 1)
            | (ids // plane == 0)
            | (ids // plane == nz - 1)
        )
        v[boundary] = np.where(v[boundary] == 0, 1.0, np.abs(v[boundary]))
        return v

    stride = 4 if math.prod(ns) > DENSE_SAMPLES else 1
    coarse = [np.r_[np.arange(0, n - 1, stride), n - 1] for n in ns]
    counts = [len(a) - 1 for a in coarse]
    if stride > 1:
        zz, yy, xx = np.meshgrid(coarse[2], coarse[1], coarse[0], indexing="ij")
        cv = sample((zz * plane + yy * nx + xx).ravel()).reshape(zz.shape)
        corners = [
            cv[
                k : cv.shape[0] - 1 + k,
                j : cv.shape[1] - 1 + j,
                i : cv.shape[2] - 1 + i,
            ]
            for k, j, i in itertools.product((0, 1), repeat=3)
        ]
        centres = [(a[:-1] + a[1:]) // 2 for a in coarse]
        zz, yy, xx = np.meshgrid(centres[2], centres[1], centres[0], indexing="ij")
        centre = sample((zz * plane + yy * nx + xx).ravel()).reshape(zz.shape)
        lo = np.minimum.reduce(corners + [centre])
        hi = np.maximum.reduce(corners + [centre])
        active = ((lo <= 0) & (hi >= 0)) | (
            np.minimum(np.abs(lo), np.abs(hi)) <= 2 * (hi - lo)
        )
        # Complete all mixed-sign fine faces of the selected band in batches.
        while True:
            additions = np.zeros_like(active)
            for axis in range(3):
                for step in (-1, 1):
                    neighbour = np.roll(active, -step, axis=2 - axis)
                    candidate = active & ~neighbour
                    sl = [slice(None)] * 3
                    sl[2 - axis] = -1 if step == 1 else 0
                    candidate[tuple(sl)] = False
                    blocks = np.argwhere(candidate)[:, ::-1]
                    for start in range(0, len(blocks), 4096):
                        bb = blocks[start : start + 4096]
                        if not len(bb):
                            continue
                        offsets = np.array(list(itertools.product(range(5), repeat=2)))
                        other = [a for a in range(3) if a != axis]
                        idx = np.empty((len(bb), 25, 3), dtype=np.int64)
                        idx[:, :, axis] = coarse[axis][bb[:, axis] + (step == 1), None]
                        for c, a in enumerate(other):
                            idx[:, :, a] = np.minimum(
                                coarse[a][bb[:, a], None] + offsets[None, :, c],
                                coarse[a][bb[:, a] + 1, None],
                            )
                        vals = sample(
                            (
                                idx[:, :, 2] * plane + idx[:, :, 1] * nx + idx[:, :, 0]
                            ).ravel()
                        ).reshape(-1, 25)
                        hit = bb[(vals.min(1) < 0) & (vals.max(1) >= 0)].copy()
                        hit[:, axis] += step
                        additions[tuple(hit[:, ::-1].T)] = True
            additions &= ~active
            if not additions.any():
                break
            active |= additions
    else:
        active = None

    offsets = np.array((0, 1, nx + 1, nx, plane, plane + 1, plane + nx + 1, plane + nx))
    triangles = []
    samples = 0
    total_faces = 0
    prev_ids = np.empty(0, dtype=np.int64)
    prev_vals = np.empty(0)
    slab_depth = max(1, min(8, 500_000 // plane))
    for k in range(0, nz - 1, slab_depth):
        end = min(nz - 1, k + slab_depth)
        bases = []
        for z in range(k, end):
            if active is None:
                jj, ii = np.meshgrid(
                    np.arange(ny - 1), np.arange(nx - 1), indexing="ij"
                )
            else:
                mask = active[z // stride]
                mask = np.repeat(
                    np.repeat(mask, np.diff(coarse[1]), axis=0),
                    np.diff(coarse[0]),
                    axis=1,
                )
                jj, ii = np.nonzero(mask)
            bases.append(z * plane + jj.ravel() * nx + ii.ravel())
        ids = np.concatenate(bases)[:, None] + offsets
        unique, inverse = np.unique(ids, return_inverse=True)
        shared = np.isin(unique, prev_ids, assume_unique=True)
        vals = np.empty(len(unique))
        vals[shared] = prev_vals[np.searchsorted(prev_ids, unique[shared])]
        samples += np.count_nonzero(~shared)
        if samples > MAX_SAMPLES:
            raise ValueError(
                "field: surface band exceeds samples budget; increase cell"
            )
        vals[~shared] = sample(unique[~shared])
        upper = unique >= end * plane
        prev_ids, prev_vals = unique[upper], vals[upper]
        values = vals[inverse].reshape(ids.shape)
        crossing = (values.min(1) < 0) & (values.max(1) >= 0)
        ids, values = ids[crossing], values[crossing]
        for tet, table in zip(_TETS, _CASES):
            ti, tv = ids[:, tet], values[:, tet]
            codes = (tv < 0).astype(np.uint8) @ np.array([1, 2, 4, 8], dtype=np.uint8)
            for code in range(1, 15):
                rows = ti[codes == code]
                if not len(rows):
                    continue
                vv = tv[codes == code]
                for pattern in table[code]:
                    pairs = rows[:, pattern]
                    # Winding follows the linear tetrahedron, before refinement.
                    va, vb = vv[:, pattern[:, 0]], vv[:, pattern[:, 1]]
                    zero = np.where(
                        va == 0,
                        rows[:, [a for a, b in pattern]],
                        np.where(vb == 0, rows[:, [b for a, b in pattern]], -1),
                    )
                    pairs = np.sort(pairs, axis=2)
                    pairs = np.where((zero >= 0)[:, :, None], zero[:, :, None], pairs)
                    triangles.append(pairs)
                    total_faces += len(pairs)
                    if total_faces > MAX_TRIANGLES:
                        raise ValueError(
                            "field: surface exceeds triangles budget; increase cell"
                        )
    if not triangles:
        raise ValueError("field: empty surface; decrease cell or change density")
    pairs = np.concatenate(triangles)
    del triangles
    flat = pairs.reshape(-1, 2)
    grid_size = math.prod(ns)
    keys, inv = np.unique(flat[:, 0] * grid_size + flat[:, 1], return_inverse=True)
    edges = np.column_stack((keys // grid_size, keys % grid_size))
    zero_vertices = np.flatnonzero(edges[:, 0] == edges[:, 1])
    del keys, flat
    faces = inv.reshape(-1, 3)
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    ]
    del pairs, inv
    verts = np.empty((len(edges), 3))
    for start in range(0, len(edges), 65536):
        e = edges[start : start + 65536]
        p, q = points(e[:, 0]), points(e[:, 1])
        va, vb = sample(e[:, 0]), sample(e[:, 1])
        scale = np.maximum(np.abs(va), np.abs(vb))
        scale[scale == 0] = 1
        fl, fr = va / scale, vb / scale
        t = np.divide(fl, fl - fr, out=np.zeros_like(fl), where=fl != fr)
        interior = np.all(
            (p > np.array([a[0] for a in axes]))
            & (q > np.array([a[0] for a in axes]))
            & (p < np.array([a[-1] for a in axes]))
            & (q < np.array([a[-1] for a in axes])),
            axis=1,
        ) & (e[:, 0] != e[:, 1])
        left = np.zeros(len(e))
        right = np.ones(len(e))
        side = np.full(len(e), -1)
        running = interior.copy()
        for _ in range(12):
            idx = np.flatnonzero(running)
            if not len(idx):
                break
            f = evaluate(p[idx] + t[idx, None] * (q[idx] - p[idx])) / scale[idx]
            running[idx[np.abs(f) < 1e-6]] = False
            same = (f < 0) == (fl[idx] < 0)
            a, b = idx[same], idx[~same]
            fr[a] *= np.where(side[a] == 0, 0.5, 1)
            fl[b] *= np.where(side[b] == 1, 0.5, 1)
            left[a] = t[a]
            fl[a] = f[same]
            side[a] = 0
            right[b] = t[b]
            fr[b] = f[~same]
            side[b] = 1
            idx = np.flatnonzero(running)
            t[idx] = left[idx] + (right[idx] - left[idx]) * fl[idx] / (
                fl[idx] - fr[idx]
            )
        # A fixed fraction is insufficient far from the origin: distinct roots
        # on incident edges can round onto the same float32 lattice corner.
        # Leave several coordinate ULPs on every varying axis, using the same
        # guard for both ends of each shared edge. Keep ordinary roots unchanged.
        delta = np.abs(q - p)
        guard = np.max(np.divide(
            4 * np.finfo(np.float32).eps * np.maximum(np.abs(p), np.abs(q)),
            delta, out=np.zeros_like(delta), where=delta != 0,
        ), axis=1)
        guard = np.maximum(1e-5, guard)
        if np.any(guard >= .5):
            raise ValueError("field: cell is below mesh coordinate precision")
        t = np.where(e[:, 0] == e[:, 1], 0, np.clip(t, guard, 1 - guard))
        verts[start : start + len(e)] = p + t[:, None] * (q - p)
    del edges
    # Vectorized oriented incidence, including exact-level singularities.
    ee = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    ordered = np.sort(ee, axis=1)
    _, inverse, counts = np.unique(
        ordered[:, 0] * len(verts) + ordered[:, 1],
        return_inverse=True,
        return_counts=True,
    )
    balance = np.bincount(inverse, weights=np.where(ee[:, 0] < ee[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(balance != 0):
        raise ValueError("field: singular surface at level; change level or cell")
    del inverse, counts, balance, ordered
    # Exact-level lattice points must have one circular incident fan.
    if len(zero_vertices):
        row, corner = np.nonzero(np.isin(faces, zero_vertices))
        order = np.argsort(faces[row, corner], kind="stable")
        row, corner = row[order], corner[order]
        vertex = faces[row, corner]
        starts = np.r_[0, np.flatnonzero(np.diff(vertex)) + 1, len(vertex)]
        for first, last in zip(starts[:-1], starts[1:]):
            rr, cc = row[first:last], corner[first:last]
            fan = dict(
                zip(faces[rr, (cc + 1) % 3].tolist(), faces[rr, (cc + 2) % 3].tolist())
            )
            if not fan:
                continue
            start = next(iter(fan))
            seen = set()
            current = start
            while current not in seen and current in fan:
                seen.add(current)
                current = fan[current]
            if current != start or len(seen) != len(fan):
                raise ValueError(
                    "field: singular vertex at level; change level or cell"
                )
    if min_component_cells:
        labels = np.arange(len(verts))
        while True:
            old = labels.copy()
            np.minimum.at(labels, ee[:, 0], labels[ee[:, 1]])
            np.minimum.at(labels, ee[:, 1], labels[ee[:, 0]])
            labels = labels[labels]
            if np.array_equal(labels, old):
                break
        roots, comp = np.unique(labels, return_inverse=True)
        low = np.full((len(roots), 3), np.inf)
        high = -low.copy()
        np.minimum.at(low, comp, verts)
        np.maximum.at(high, comp, verts)
        keep = np.max(high - low, axis=1) >= min_component_cells * cell
        # Always retain the largest component, even for a sub-cell object.
        keep[np.argmax(np.bincount(comp))] = True
        faces = faces[keep[comp[faces[:, 0]]]]
    del ee
    used, inv = np.unique(faces, return_inverse=True)
    verts = verts[used]
    faces = inv.reshape(-1, 3)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(len(faces), dtype=np.int32) * 3)
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    mesh.update()
    # bpy versions differ in whether an all-smooth mesh stores this attribute.
    sharp = mesh.attributes.get("sharp_face")
    if sharp is not None:
        mesh.attributes.remove(sharp)
    for tag in ("field", f"{name}.field"):
        attr = mesh.attributes.new(_core.FACE_TAG + tag, "BOOLEAN", "FACE")
        attr.data.foreach_set("value", np.ones(len(faces), dtype=bool))
    co = np.empty(len(verts) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    area = np.empty(len(faces))
    mesh.polygons.foreach_get("area", area)
    if not np.all(np.isfinite(co)) or np.any(area == 0):
        bpy.data.meshes.remove(mesh)
        raise ValueError(
            "field: degenerate triangles at mesh precision; change level or cell"
        )
    ob = _core.link(bpy.data.objects.new(name, mesh))
    ob[_core.NAME] = name
    return ob
