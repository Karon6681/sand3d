"""A spoon and a fork, from generic verbs — the regression test for "the vocabulary has a
hole": MeshCoder had to hard-code both.

Millimetres; the piece lies along +X from its business end, Z up. The handle is shared:
a loft of rounded sections along a gently rising line, narrow at the neck and widening
to a rounded end.
"""
import math

import sand3d as s3d


def handle(length=135, neck=(8.5, 2.6), end=(24, 3.2), rise=16, start=(0, 0, 0), stations=7, end_steps=6):
    """Sections from the neck to the end of the handle, square to its line; the last
    ones close it on a half circle in plan, so the end is round."""
    x0, y0, z0 = start
    r = end[0] / 2
    body = length - r                      # where the round end begins

    def place(x, width, thick):
        u = (x - x0) / length
        return s3d.at(s3d.superellipse(width, thick, 4), x=x, y=y0, z=z0 + rise * (3 * u ** 2 - 2 * u ** 3), rz=90)
    sections = []
    for k in range(stations):
        u = k / (stations - 1)
        sections.append(place(x0 + body * u, neck[0] + (end[0] - neck[0]) * u ** 1.6,
                              neck[1] + (end[1] - neck[1]) * u))
    for k in range(1, end_steps + 1):
        a = math.radians(85 * k / end_steps)
        sections.append(place(x0 + body + r * math.sin(a), end[0] * math.cos(a), end[1] * (1 - 0.3 * k / end_steps)))
    return s3d.loft("handle", *sections, follow=True)


def spoon(bowl=(62, 42), depth=13, wall=1.4, rim_r=0.5):
    """A bowl — half an ellipsoid, hollowed, its rim softened — and the handle."""
    length, width = bowl
    cup = s3d.scale(s3d.sphere("bowl", 1), (length, width, 2 * depth))
    s3d.split(cup, s3d.Frame((0, 0, depth)), keep="below")
    cup = s3d.at(cup, z=-depth)                                        # its rim on z = 0
    s3d.shell(cup, wall, open="cut")
    s3d.fillet(cup, rim_r, edges="shell")
    return s3d.join(cup, s3d.at(handle(start=(length / 2 - 3, 0, -wall / 2)), z=0))


def fork(head=(48, 25), tines=4, gap=2.2, root=17, t=2.2, bend_r=140):
    """A flat head cut into `tines` from its tip back to `root`, bent to a curve by
    keeping what a curved plate shares with it — and the handle."""
    length, width = head
    outline = s3d.poly([(-length, -width / 2), (0, -width / 2 + 3), (8, -4.5), (8, 4.5), (0, width / 2 - 3),
                        (-length, width / 2)], fillet=[4, 0, 3, 3, 0, 4])
    pitch = width / tines
    for k in range(1, tines):
        y = -width / 2 + k * pitch
        outline = outline - s3d.at(s3d.slot(2 * (length - root) + gap, gap), x=-length, y=y)
    prism = s3d.extrude("head", outline, 60, both=True)
    steps = 24
    curve = [(bend_r * math.sin(a), 0, bend_r * (1 - math.cos(a)))       # through the neck, rising to the tips
             for a in (-0.42 + 0.5 * k / steps for k in range(steps + 1))]
    plate = s3d.sweep("bend", s3d.rect(t, 4 * width), curve)
    s3d.intersect(prism, plate)
    return s3d.join(prism, handle(start=(6, 0, -t / 4)))
