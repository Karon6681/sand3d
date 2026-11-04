"""The weight stack of a cable machine.

A column of plates resting in two rubber bumpers, two guide rods standing over the
top plate, and the selector: two nuts and a stem on top, the pin's head at the
chosen plate and a spring lanyard hanging from under the nuts down to it.

Millimetres. The stack's frame has its origin under the middle of the lowest
plate, X along the plates, -Y towards the user.
"""
import json
import sys

import sand3d as s3d


# ---------------------------------------------------------------- the object: architecture first, then details

def weight_stack(n_plates=14, selected=9):
    """The plates, then each detail on their anchors. Any `add_` line can be deleted:
    the rest of the object does not change."""
    stack = plate_column(n_plates)
    add_bumpers(stack)
    add_guide_rods(stack)
    add_selector(stack, selected)
    return stack.root


def plate_column(n_plates=14, pitch=24, plate_h=22, plate_d=100, guide_spacing=300):
    """`n_plates` plates `pitch` apart. Publishes where details go: every plate's
    height, the top face, the front face and the two guide lines."""
    bottoms = [i * pitch for i in range(n_plates)]
    plate_z = [z + plate_h / 2 for z in bottoms]
    return s3d.assembly("weight_stack",
                        *[s3d.at(weight_plate(d=plate_d, h=plate_h), z=z) for z in bottoms],
                        plate_z=plate_z, top=bottoms[-1] + plate_h, front=-plate_d / 2,
                        guides=(-guide_spacing / 2, guide_spacing / 2))


# ---------------------------------------------------------------- details, each on the stack's anchors

def add_bumpers(stack, sink=10):
    """Two rubber cups under the guide lines, taking the lowest plate `sink` deep."""
    s3d.attach(stack, "bumpers", *[s3d.at(bumper(), x=x, z=sink, anchor="top") for x in stack.guides])


def add_guide_rods(stack, gap=8):
    """Two rods on the guide lines, standing `gap` above the top plate."""
    s3d.attach(stack, "guide_rods", *[s3d.at(guide_rod(), x=x, z=stack.top + gap, anchor="bottom")
                                      for x in stack.guides])


def add_selector(stack, selected=9, nut_t=6, nut_gap=4, head_out=12, lanyard_y=-28):
    """On the top plate, two nuts and the stem; the lanyard's loop sits under the
    upper nut and its spring hangs in front, down to the pin head at plate `selected`."""
    upper = stack.top + nut_t + nut_gap     # under the upper nut
    pinned = stack.plate_z[selected]
    s3d.attach(stack, "selector",
               s3d.at(hex_nut(t=nut_t), z=stack.top, anchor="bottom"),
               s3d.at(hex_nut(t=nut_t), z=upper, anchor="bottom"),
               s3d.at(selector_stem(), z=stack.top),
               s3d.at(pin_head(), y=stack.front - head_out, z=pinned, rx=90),
               s3d.at(spring_lanyard(length=upper - pinned), y=lanyard_y, z=(upper + pinned) / 2, rz=-90))


# ---------------------------------------------------------------- parts, in the order they are called

def weight_plate(w=420, d=100, h=22, hole_d=16, edge_r=2):
    """A steel plate with softened edges, drilled front to back for the selector pin."""
    plate = s3d.fillet(s3d.box("weight_plate", w, d, h), edge_r)
    return s3d.hole(plate, hole_d, on=plate.face("-Y").move(y=h / 2))


def bumper(d_foot=50, d_mouth=46, h=60, wall=2):
    """A rubber cup the stack rests in, open at the top."""
    cup = s3d.revolve("bumper", [(0, -h / 2), (d_foot / 2, -h / 2), (d_mouth / 2, h / 2), (0, h / 2)])
    return s3d.shell(cup, wall, open="top")


def guide_rod(d=25, length=900, collar_d=35, collar_h=20):
    """A round rod with a collar at its foot."""
    foot, head = -length / 2, length / 2
    return s3d.revolve("guide_rod", [(0, foot), (collar_d / 2, foot), (collar_d / 2, foot + collar_h),
                                     (d / 2, foot + collar_h), (d / 2, head), (0, head)])


def hex_nut(across_corners=32, hole_d=16.5, t=6):
    return s3d.extrude("hex_nut", s3d.polygon(6, across_corners) - s3d.circle(hole_d), t, both=True)


def selector_stem(d=15, h=30, wall=1.5):
    """The top of the stem that threads the plates, an open sleeve."""
    return s3d.tube("selector_stem", d, d - 2 * wall, h)


def pin_head(d=36, t=3, boss_d=12, boss_t=6, hole_d=8):
    """The head of the selector pin, a disc with a boss, facing the user."""
    return s3d.revolve("pin_head", [(hole_d / 2, -boss_t / 2), (boss_d / 2, -boss_t / 2), (boss_d / 2, boss_t / 2 - t),
                                    (d / 2, boss_t / 2 - t), (d / 2, boss_t / 2), (hole_d / 2, boss_t / 2)])


def spring_lanyard(length=150, loop_d=44, loop_gap=40, neck=40, coil_x=24, lean=12, coil_d=4, wire_d=2.2,
                   pitch=2.6):
    """A loop lying flat on top, round the selector stem and open by `loop_gap`
    degrees, and a close-wound spring hanging `length` down from it, `neck` away
    from the loop's centre and leaning `lean` outwards by its foot."""
    top = length / 2 - wire_d / 2
    loop = s3d.arc((coil_x - neck, 0, top), loop_d / 2, loop_gap / 2, 360 - loop_gap / 2)
    hanging = s3d.at(s3d.helix(coil_d, pitch, 2 * top), x=coil_x, z=top, rx=180)   # from the loop, down
    coil = [(x + lean * (top - z) / (2 * top), y, z) for x, y, z in hanging.points]
    return s3d.wire("spring_lanyard", list(loop.points) + coil, wire_d)


if __name__ == "__main__":
    s3d.empty_scene()
    s3d.export(s3d.group("weight_stack", weight_stack(**json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}"))),
               sys.argv[1] if len(sys.argv) > 1 else "weight_stack.glb")
