"""Joints: declaring them, the kinematic tree, copies, transforms and mirrors,
and every error a joint can raise at declaration."""
import math
import unittest

import bpy  # noqa: F401

import sand3d as s3d
from sand3d import _core
from sand3d.joints import axis_in_world, joints_of, parent_of


def close(test, a, b, places=6):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        test.assertAlmostEqual(x, y, places=places)


def washer():
    """A cabinet with a door on the -Y edge of its +X face."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.at(s3d.box("door", 20, 560, 800), x=310, z=25)
    return cab, door


class Declaring(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab, self.door = washer()

    def test_a_joint_returns_its_child_and_neither_moves_nor_reparents_it(self):
        before = _core.world(self.door)
        got = s3d.joint("door_hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 143))
        self.assertIs(got, self.door)
        self.assertEqual(_core.world(self.door), before)
        self.assertIsNone(self.door.parent)
        self.assertIs(parent_of(self.door), self.cab)
        j, = joints_of(self.door)
        self.assertEqual((j["name"], j["kind"], j["range"], j["value"]), ("door_hinge", "revolute", [0.0, 143.0], 0))
        origin, axis = axis_in_world(self.door, 0)
        close(self, origin, (300, -300, 425))
        close(self, axis, (0, 0, -1))

    def test_the_axis_is_a_direction_of_the_parent_s_space_or_the_frame_s_normal(self):
        knob = s3d.at(s3d.cylinder("knob", 40, 10), on=self.cab.face("+X").move(x=200, y=700))
        s3d.joint("dial", knob, self.cab.face("+X"), kind="continuous")
        close(self, axis_in_world(knob, 0)[1], (1, 0, 0))                # "normal": the face's Z
        unit = s3d.transform(s3d.group("unit", self.cab, self.door, knob), rz=90)
        s3d.joint("hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 90))
        close(self, axis_in_world(self.door, 0)[1], (0, 0, -1))         # in the unit's space: Z stays Z
        drawer = s3d.at(s3d.box("drawer", 100, 300, 50), x=200, z=700)
        unit2 = s3d.transform(s3d.group("unit2", s3d.copy(self.cab), drawer), rx=90)
        s3d.joint("slide", drawer, s3d.bbox(unit2.children[0], "+X"), kind="prismatic", axis="+X", range=(0, 200))
        close(self, axis_in_world(drawer, 0)[1], (1, 0, 0))
        self.assertIsNotNone(unit)

    def test_a_group_or_an_assembly_is_a_part(self):
        handle = s3d.at(s3d.box("handle", 30, 20, 120), x=330, y=200, z=425)
        leaf = s3d.group("leaf", self.door, handle)
        s3d.joint("hinge", leaf, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 90))
        self.assertEqual(joints_of(leaf)[0]["name"], "hinge")
        arm = s3d.assembly("arm", s3d.box("body", 10, 10, 100), tip=s3d.Frame((0, 0, 100)))
        got = s3d.joint("shoulder", arm, self.cab.face("top"), axis="+X", range=(-45, 45))
        self.assertIs(got, arm)
        lamp = s3d.at(s3d.box("lamp", 20, 20, 20), z=950)
        s3d.joint("elbow", lamp, arm.tip, axis="+Y", range=(-90, 90))   # an assembly's anchor is read on it
        self.assertIs(parent_of(lamp), arm.root)

    def test_joints_may_come_after_the_group(self):
        machine = s3d.assembly("machine", self.cab, self.door)
        s3d.joint("hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 90))
        self.assertIs(self.door.parent, machine.root)
        close(self, axis_in_world(self.door, 0)[0], (300, -300, 425))

    def test_two_joints_on_one_child_hang_on_one_parent(self):
        cap = s3d.at(s3d.cylinder("cap", 30, 20), z=850)
        top = self.cab.face("top")
        s3d.joint("turn", cap, top, kind="revolute", range=(0, 720))
        s3d.joint("lift", cap, top, kind="prismatic", follows="turn", ratio=3 / 360)
        self.assertEqual([j["name"] for j in joints_of(cap)], ["turn", "lift"])
        self.assertEqual(joints_of(cap)[1]["follows"], 0)
        with self.assertRaisesRegex(ValueError, r"^joint 'tilt': 'cap' already hangs on 'cab' by 'turn': every joint "
                                                r"of one child is read on the same parent$"):
            s3d.joint("tilt", cap, self.door.face("+X"), range=(0, 1))

    def test_the_frame_stays_on_its_part(self):
        hinge = s3d.bbox(self.cab, "+X-Y")
        s3d.transform([self.cab, self.door], x=1000)        # read, then moved: the pivot is where the part is
        s3d.joint("hinge", self.door, hinge, axis="-Z", range=(0, 90))
        close(self, axis_in_world(self.door, 0)[0], (1300, -300, 425))


class CopiesAndMirrors(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab, self.door = washer()
        s3d.joint("hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 143))

    def test_a_copy_carries_its_joints_and_hangs_on_the_same_parent(self):
        twin = s3d.copy(self.door)
        self.assertEqual(joints_of(twin), joints_of(self.door))
        self.assertIs(parent_of(twin), self.cab)

    def test_a_copy_of_the_whole_hangs_on_the_copy_of_its_parent(self):
        machine = s3d.group("machine", self.cab, self.door)
        twin = s3d.copy(machine)
        door2 = next(c for c in twin.children if c.get("s3d_name") == "door")
        cab2 = next(c for c in twin.children if c.get("s3d_name") == "cab")
        self.assertIs(parent_of(door2), cab2)
        self.assertIs(parent_of(self.door), self.cab)
        copies = s3d.at(machine, on=s3d.grid(2, 1, 2000))
        for copy in copies:
            door = next(c for c in copy.children if c.get("s3d_name") == "door")
            self.assertIs(parent_of(door).parent, copy)

    def test_transform_moves_a_jointed_part_with_its_pivot(self):
        s3d.transform(self.door, x=50)
        close(self, axis_in_world(self.door, 0)[0], (350, -300, 425))
        from sand3d.joints import rest_basis
        self.assertEqual(rest_basis(self.door).translation.x, 360.0)

    def test_a_mirror_image_moves_as_the_image_of_the_motion(self):
        twin = s3d.transform(s3d.copy(self.door), mirror="y")
        origin, axis = axis_in_world(twin, 0)
        close(self, origin, (300, 300, 425))
        close(self, axis, (0, 0, 1))                       # −M·u: the revolute axis turns over
        self.assertEqual(joints_of(twin)[0]["range"], [0.0, 143.0])
        drawer = s3d.at(s3d.box("drawer", 100, 300, 50), x=-200, z=700)
        s3d.joint("slide", drawer, s3d.bbox(self.cab, "-X"), kind="prismatic", axis="-X", range=(0, 200))
        image = s3d.transform(s3d.copy(drawer), mirror="x")
        close(self, axis_in_world(image, 0)[1], (1, 0, 0))   # M·u: a slide's image slides the image way


class Errors(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab, self.door = washer()
        self.edge = s3d.bbox(self.cab, "+X-Y")

    def raises(self, message, error=ValueError, **kwargs):
        args = dict(name="j", child=self.door, on=self.edge, axis="-Z", range=(0, 90))
        args.update(kwargs)
        if args["axis"] is None:
            del args["axis"]
        with self.assertRaises(error) as caught:
            s3d.joint(**args)
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(joints_of(self.door), [])

    def test_an_unknown_kind_or_axis(self):
        self.raises("joint 'j': kind 'hinge' is none of revolute, continuous, prismatic, free", kind="hinge")
        self.raises("joint 'j': axis 'Z+' is none of \"+X\" \"-X\" \"+Y\" \"-Y\" \"+Z\" \"-Z\" — a direction in the "
                    "space the parent lives in — or \"normal\", the Z of the frame `on`", axis="Z+")
        self.raises("joint 'j': axis (0, 0, 1) is none of \"+X\" \"-X\" \"+Y\" \"-Y\" \"+Z\" \"-Z\" — a direction in "
                    "the space the parent lives in — or \"normal\", the Z of the frame `on`", axis=(0, 0, 1))

    def test_a_range_missing_or_given_where_it_has_no_place(self):
        self.raises("joint 'j': a revolute joint needs range=(low, high), in degrees — kind=\"continuous\" turns "
                    "without end", range=None)
        self.raises("joint 'j': a prismatic joint needs range=(low, high), in the program's unit", kind="prismatic",
                    range=None)
        self.raises("joint 'j': a continuous joint turns without end: it takes no range", kind="continuous")
        self.raises("joint 'j': a free joint moves freely: it takes no range", kind="free", axis=None)
        s3d.joint("lead", s3d.at(s3d.box("k", 5, 5, 5), z=860), self.cab.face("top"), kind="continuous")
        self.raises("joint 'j': a joint that follows another takes no range: its range is the other's × ratio",
                    follows="lead")

    def test_a_range_without_zero_or_empty(self):
        self.raises("joint 'j': range (10, 90) does not contain 0: everything is built at rest, every joint at 0 — "
                    "build the part where it rests, and give the range from there", range=(10, 90))
        self.raises("joint 'j': range (90, 0) is empty: low comes before high", range=(90, 0))
        self.raises("joint 'j': range is (low, high), two numbers, not 90", TypeError, range=90)

    def test_an_axis_given_to_a_free_joint(self):
        self.raises("joint 'j': a free joint moves every way: it takes no axis", kind="free", range=None)
        self.raises("joint 'j': a free joint moves every way: it takes no axis", kind="free", range=None,
                    axis="normal")
        plug = s3d.at(s3d.box("plug", 10, 10, 10), x=400)
        s3d.joint("cord", plug, self.cab.face("+X"), kind="free")       # no axis written: none taken
        self.assertIsNone(joints_of(plug)[0]["axis"])

    def test_on_not_read_on_a_part_or_read_on_the_child(self):
        self.raises("joint 'j': on is a frame built by hand, read on no part: read it on the part the child moves on "
                    "— ob.face(...), s3d.bbox(ob, \"+X-Y\"), an assembly's anchor — so that the joint knows its parent",
                    on=s3d.Frame((300, -300, 425)))
        self.raises("joint 'j': on is a frame read on the parent part — ob.face(...), s3d.bbox(ob, ...), an "
                    "assembly's anchor — not tuple", TypeError, on=(300, -300, 425))
        self.raises("joint 'j': on is read on the child itself: read it on the part 'door' moves on",
                    on=self.door.face("+X"))
        knob = s3d.at(s3d.box("knob", 10, 10, 10), x=325)
        leaf = s3d.group("leaf", self.door, knob)
        with self.assertRaisesRegex(ValueError, r"^joint 'j': on is read on 'knob', which is inside the child: read it "
                                                r"on the part 'leaf' moves on$"):
            s3d.joint("j", leaf, knob.face("+X"), range=(0, 1))

    def test_a_cycle_in_the_tree(self):
        s3d.joint("hinge", self.door, self.edge, axis="-Z", range=(0, 90))
        with self.assertRaisesRegex(ValueError, r"^joint 'back': 'door' already hangs on 'cab', through the joints "
                                                r"declared: hanging 'cab' on it makes a cycle$"):
            s3d.joint("back", self.cab, self.door.face("+X"), range=(0, 1))
        knob = s3d.at(s3d.box("knob", 10, 10, 10), x=325, z=500)
        s3d.joint("turn", knob, self.door.face("+X"), kind="continuous")
        with self.assertRaisesRegex(ValueError, r"^joint 'loop': 'knob' already hangs on 'cab'"):
            s3d.joint("loop", self.cab, knob.face("+X"), range=(0, 1))

    def test_the_name_child_and_ratio(self):
        with self.assertRaisesRegex(TypeError, r"^joint: name is the joint's name, a string, not 3$"):
            s3d.joint(3, self.door, self.edge, range=(0, 1))
        with self.assertRaisesRegex(TypeError, r"^joint 'j': child is the part that moves — a solid, a group or an "
                                               r"assembly — not Frame$"):
            s3d.joint("j", self.edge, self.edge, range=(0, 1))
        s3d.joint("lead", s3d.at(s3d.box("k", 5, 5, 5), z=860), self.cab.face("top"), kind="continuous")
        self.raises("joint 'j': ratio is a number other than 0 — the follower's value per unit of the followed "
                    "joint's — not 0", range=None, follows="lead", ratio=0)


class Follows(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.base = s3d.box("base", 200, 100, 10)
        self.big = s3d.at(s3d.cylinder("big", 60, 10), x=-40, z=10)
        self.small = s3d.at(s3d.cylinder("small", 20, 10), x=0, z=10)
        s3d.joint("drive", self.big, self.base.face("top").move(-40), kind="continuous")

    def test_follows_a_child_or_a_joint_name(self):
        s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows=self.big, ratio=-3)
        self.assertEqual(joints_of(self.small)[0]["ratio"], -3.0)
        self.assertIs(self.small["s3d~follows.0"], self.big)
        third = s3d.at(s3d.cylinder("third", 10, 10), x=30, z=10)
        s3d.joint("idler", third, self.base.face("top").move(30), kind="continuous", follows="gear")
        self.assertIs(third["s3d~follows.0"], self.small)

    def test_an_ambiguous_name_a_cycle_a_free_leader(self):
        s3d.copy(self.big)
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows 'drive', which names 2 joints \(on 'big': "
                                                r"copies carry their joints\): give the part it moves, follows=part$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows="drive")
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows 'nope', and no joint has that name$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows="nope")
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows its own child, which has no joint yet: a "
                                                r"joint cannot follow itself \(a cycle\)$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows=self.small)
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows 'base', which has no joint$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows=self.base)
        loose = s3d.at(s3d.box("loose", 5, 5, 5), x=90, z=10)
        s3d.joint("float", loose, self.base.face("top"), kind="free")
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows 'float', a free joint: it has no value to "
                                                r"follow$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows="float")
        with self.assertRaisesRegex(ValueError, r"^joint 'f': a free joint follows nothing: it has no value$"):
            s3d.joint("f", self.small, self.base.face("top"), kind="free", follows="float")

    def test_two_joints_followed_by_their_child_need_the_name(self):
        s3d.joint("lift", self.big, self.base.face("top").move(-40), kind="prismatic", axis="+Z", range=(0, 5))
        with self.assertRaisesRegex(ValueError, r"^joint 'gear': follows 'big', which has 2 joints \('drive', 'lift'\): "
                                                r"name the one, follows=\"drive\"$"):
            s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows=self.big)


    def test_a_slide_that_follows_an_endless_turn(self):
        rack = s3d.at(s3d.box("rack", 100, 10, 10), y=40, z=10)
        message = (r"^joint 'slide': a slide that follows '{}', which turns without end, would slide without end: "
                   r"give '{}' a range$")
        with self.assertRaisesRegex(ValueError, message.format("drive", "drive")):
            s3d.joint("slide", rack, self.base.face("top"), kind="prismatic", axis="+X", follows=self.big, ratio=0.5)
        s3d.joint("gear", self.small, self.base.face("top"), kind="continuous", follows=self.big, ratio=-3)
        with self.assertRaisesRegex(ValueError, message.format("gear", "gear")):           # down a chain of follows
            s3d.joint("slide", rack, self.base.face("top"), kind="prismatic", axis="+X", follows="gear")
        self.assertEqual(joints_of(rack), [])


class PlacedOn(unittest.TestCase):
    """A child placed on a frame read on its parent must still sit where that frame put
    it: its pivot follows the parent, and it would not."""

    def setUp(self):
        s3d.empty_scene()
        self.cab = s3d.box("cab", 200, 200, 200)
        self.top = self.cab.face("+Z")
        self.knob = s3d.box("knob", 80, 20, 20)

    def refused(self):
        with self.assertRaisesRegex(ValueError, r"^joint 'dial': 'knob' is no longer where its frame put it on 'cab' "
                                                r"— 'cab' moved after the frame was read, or after 'knob' was placed: "
                                                r"its pivot would follow 'cab' and 'knob' would not\. Group them so "
                                                r"they move together, or read the frame on 'cab' where it is now$"):
            s3d.joint("dial", self.knob, self.top, kind="continuous")
        self.assertEqual(joints_of(self.knob), [])

    def test_the_parent_moved_before_the_child_was_placed(self):
        s3d.at(self.cab, x=300)
        s3d.at(self.knob, on=self.top)
        self.refused()

    def test_the_parent_moved_after_the_child_was_placed(self):
        s3d.at(self.knob, on=self.top)
        s3d.transform(self.cab, x=300)
        self.refused()

    def test_moved_together_or_placed_again(self):
        s3d.at(self.knob, on=self.top)
        s3d.at(s3d.group("g", self.cab, self.knob), x=300)                  # together: still on it
        s3d.joint("dial", self.knob, self.top, kind="continuous")
        close(self, axis_in_world(self.knob, 0)[0], (300, 0, 200))
        s3d.empty_scene()
        cab = s3d.box("cab", 200, 100, 20)
        keys = s3d.at(s3d.box("key", 10, 10, 5), on=s3d.grid(3, 2, 30, 30, on=cab.face("+Z")))
        s3d.transform(s3d.group("keys", keys), z=1)                         # the child moved on purpose
        for k in keys:
            s3d.joint("press", k, cab.face("+Z"), kind="prismatic", axis="-Z", range=(0, 4))


class OtherVerbs(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab, self.door = washer()
        s3d.joint("hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 143))

    def test_at_refuses_a_jointed_part(self):
        message = ("at: 'door' moves by its joint 'hinge': placing it would tear it off its pivot — place it before "
                   "the joint, or s3d.transform it, which moves it with its pivot")
        for call in (lambda: s3d.at(self.door, x=1), lambda: s3d.at(self.door, on=s3d.grid(2, 1, 10)),
                     lambda: s3d.at([self.door], z=3)):
            with self.assertRaises(ValueError) as caught:
                call()
            self.assertEqual(str(caught.exception), message)
        s3d.at(self.door)                                   # nothing to change: nothing refused
        machine = s3d.group("machine", self.cab, self.door)
        s3d.at(machine, x=100)                              # its group moves whole

    def test_join_and_cut_refuse_to_use_up_a_jointed_part_or_its_parent(self):
        other = s3d.box("other", 10, 10, 10)
        for verb in ("join", "cut", "intersect", "merge"):
            with self.subTest(verb=verb):
                call = (lambda: getattr(s3d, verb)(other, self.door)) if verb != "merge" else \
                    (lambda: s3d.merge(other, self.door))
                with self.assertRaises(ValueError) as caught:
                    call()
                self.assertEqual(str(caught.exception),
                                 f"{verb}: 'door' moves by its joint 'hinge': {verb} would use it up, and the joint with "
                                 f"it (a copy carries its joints too) — make the tool from a part that does not move; "
                                 f"s3d.seat(host, part) cuts the cavity and leaves the part as it is")
        with self.assertRaisesRegex(ValueError, r"^join: 'door' hangs on 'cab' by 'hinge': join would use 'cab' up, "
                                                r"and leave the joint on nothing$"):
            s3d.join(other, self.cab)
        s3d.cut(self.door, s3d.at(s3d.cylinder("peephole", 10, 40), x=310, z=500, ry=90))   # the jointed part stays
        with self.assertRaisesRegex(ValueError, r"^join: 'door' moves by its joint 'hinge'"):
            s3d.join(self.cab, s3d.copy(self.door))

    def test_the_digest_holds_the_joint(self):
        from harness import _object_digest
        before = _object_digest(self.door)
        self.door["s3d~joints"] = self.door["s3d~joints"].replace("143.0", "140.0")
        self.assertNotEqual(_object_digest(self.door), before)


if __name__ == "__main__":
    unittest.main()
