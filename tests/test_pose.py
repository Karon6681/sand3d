"""`pose`: forward kinematics down the kinematic tree, several joints on one
child, followers, mirrors and copies, and every error `pose` raises."""
import math
import unittest

import bpy  # noqa: F401
from mathutils import Matrix, Vector

import sand3d as s3d
from sand3d import _core
from sand3d.joints import joints_of


def at(ob):
    return _core.world(ob).translation


def close(test, a, b, places=3):
    test.assertEqual(len(a), len(b))
    for x, y in zip(a, b):
        test.assertAlmostEqual(x, y, places=places)


class Washer(unittest.TestCase):
    """A door on the -Y edge of a +X face, a handle in its group, a knob on the door."""

    def setUp(self):
        s3d.empty_scene()
        self.cab = s3d.box("cab", 600, 600, 850)
        self.door = s3d.at(s3d.box("door", 20, 560, 800), x=310, z=25)
        self.handle = s3d.at(s3d.box("handle", 30, 20, 100), x=335, y=200, z=400)
        self.leaf = s3d.group("leaf", self.door, self.handle)
        s3d.joint("hinge", self.leaf, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 143))
        face = self.door.face("+X").move(x=250, y=425)
        self.knob = s3d.at(s3d.cylinder("knob", 20, 10), on=face)
        s3d.joint("dial", self.knob, face, kind="continuous")
        self.root = s3d.assembly("washer", self.cab, self.leaf, self.knob)

    def test_a_door_opens_outward_about_minus_z_and_carries_its_group(self):
        got = s3d.pose(self.leaf, 90)
        self.assertIs(got, self.leaf)
        close(self, at(self.door), (600, -310, 25))       # +X turned toward -Y about -Z: outward
        close(self, at(self.handle), (800, -335, 400))
        close(self, at(self.knob), (850, -320, 425))      # jointed on the door: it goes with it
        close(self, at(self.cab), (0, 0, 0))

    def test_the_root_poses_a_joint_by_its_name(self):
        s3d.pose(self.root, 45, joint="hinge")
        self.assertEqual(joints_of(self.leaf)[0]["value"], 45.0)
        self.assertIs(s3d.pose(self.root, 30, joint="dial"), self.root)
        self.assertEqual(joints_of(self.knob)[0]["value"], 30.0)

    def test_back_at_zero_every_part_is_where_it_was_built_to_the_bit(self):
        rest = {ob.name: Matrix(ob.matrix_basis) for ob in bpy.data.objects}
        s3d.pose(self.leaf, 77.7)
        s3d.pose(self.knob, 1234.5)
        self.assertNotEqual(Matrix(self.leaf.matrix_basis), rest["leaf"])
        s3d.pose(self.knob, 0)
        s3d.pose(self.leaf, 0)
        self.assertEqual({ob.name: Matrix(ob.matrix_basis) for ob in bpy.data.objects}, rest)

    def test_a_continuous_joint_turns_without_end(self):
        s3d.pose(self.knob, 720)
        close(self, at(self.knob), (320, 250, 425))

    def test_a_jointed_part_moves_by_its_joint_whatever_group_it_is_in(self):
        lamp = s3d.at(s3d.box("lamp", 20, 20, 20), x=100, z=900)
        s3d.joint("tilt", lamp, self.cab.face("top"), axis="+X", range=(-90, 90))
        s3d.attach(s3d.Handle(self.leaf), "fixture", lamp)          # in the door's group, jointed on the cab
        s3d.pose(self.leaf, 90)
        close(self, at(lamp), (100, 0, 900))
        s3d.pose(lamp, 90)
        close(self, at(lamp), (100, -50, 850))

    def test_a_part_with_no_joint_moves_with_its_group(self):
        s3d.pose(self.leaf, 30)
        m = _core.world(self.door).inverted() @ _core.world(self.handle)
        close(self, m.translation, (25, 200, 375))

    def test_copy_and_transform_while_posed(self):
        s3d.pose(self.leaf, 60)
        twin = s3d.copy(self.leaf)
        self.assertEqual(joints_of(twin)[0]["value"], 60.0)
        door2 = next(c for c in twin.children if c.get("s3d_name") == "door")
        close(self, at(door2), at(self.door))
        s3d.transform(twin, z=1000)                               # with its pivot
        s3d.pose(twin, 0)
        close(self, at(door2), (310, 0, 1025))


class Mirrored(unittest.TestCase):

    def test_two_mirrored_leaves_open_the_same_way(self):
        s3d.empty_scene()
        w, h, post = 750, 1200, 100
        l_post = s3d.at(s3d.box("post", post, post, h + 300), x=-w - post / 2)
        r_post = s3d.transform(s3d.copy(l_post), mirror="x")
        board = s3d.at(s3d.box("board", w, 40, h), x=-w / 2, z=150)
        leaf = s3d.group("leaf", board)
        s3d.joint("hinge", leaf, s3d.bbox(l_post, "+X"), axis="+Z", range=(0, 90))
        twin = s3d.transform(s3d.copy(leaf), mirror="x")
        s3d.assembly("gate", l_post, r_post, leaf, twin)
        s3d.pose(leaf, 90)
        s3d.pose(twin, 90)
        board2 = twin.children[0]
        close(self, at(board), (-w, w / 2, 150))           # +X turned toward +Y
        close(self, at(board2), (w, w / 2, 150))           # the image turns toward +Y too
        self.assertLess(_core.world(board2).to_3x3().determinant(), 0)


class SeveralJoints(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.bottle = s3d.cylinder("bottle", 60, 200)
        self.cap = s3d.at(s3d.cylinder("cap", 40, 20), z=200)
        top = self.bottle.face("top")
        s3d.joint("cap_turn", self.cap, top, range=(0, 720))
        s3d.joint("cap_lift", self.cap, top, kind="prismatic", follows="cap_turn", ratio=3 / 360)

    def test_a_screw_is_a_turn_and_a_slide_that_follows_it(self):
        s3d.pose(self.cap, 360, joint="cap_turn")
        close(self, at(self.cap), (0, 0, 203))
        self.assertAlmostEqual(joints_of(self.cap)[1]["value"], 3.0)
        s3d.pose(self.cap, 90, joint="cap_turn")
        close(self, at(self.cap), (0, 0, 200.75))
        close(self, _core.world(self.cap).to_euler(), (0, 0, math.pi / 2))

    def test_two_joints_on_one_child_compose_in_order(self):
        s3d.empty_scene()
        base = s3d.box("base", 100, 100, 10)
        arm = s3d.at(s3d.box("arm", 100, 10, 10), x=50, z=10)
        s3d.joint("slide", arm, base.face("top"), kind="prismatic", axis="+X", range=(0, 50))
        s3d.joint("swing", arm, base.face("top"), axis="+Z", range=(-180, 180))
        s3d.pose(arm, 20, joint="slide")
        s3d.pose(arm, 90, joint="swing")                    # the turn rides on the slide: about x = 20
        close(self, at(arm), (20, 50, 10))
        s3d.pose(arm, 0, joint="slide")
        close(self, at(arm), (0, 50, 10))

    def test_a_follower_says_whom_to_pose(self):
        with self.assertRaisesRegex(ValueError, r"^pose 'cap_lift': it follows 'cap_turn' at 0\.00833333 × its value: "
                                                r"pose 'cap_turn'$"):
            s3d.pose(self.cap, 1, joint="cap_lift")
        with self.assertRaisesRegex(ValueError, r"^pose: 'cap' has 2 joints \('cap_turn', 'cap_lift'\): say which, "
                                                r"joint=\"cap_turn\"$"):
            s3d.pose(self.cap, 1)


class Followers(unittest.TestCase):

    def test_a_gear_train(self):
        s3d.empty_scene()
        base = s3d.box("base", 200, 100, 10)
        big = s3d.at(s3d.cylinder("big", 60, 10), x=-40, z=10)
        small = s3d.at(s3d.box("small", 20, 4, 10), z=10)
        third = s3d.at(s3d.box("third", 10, 2, 10), x=30, z=10)
        s3d.joint("drive", big, base.face("top").move(-40), kind="continuous")
        s3d.joint("gear", small, base.face("top"), kind="continuous", follows=big, ratio=-3)
        s3d.joint("idler", third, base.face("top").move(30), kind="revolute", follows="gear", ratio=-1 / 9)
        s3d.pose(big, 30)
        self.assertEqual([joints_of(o)[0]["value"] for o in (big, small, third)], [30.0, -90.0, 10.0])
        close(self, _core.world(small).to_euler(), (0, 0, -math.pi / 2))
        close(self, _core.world(third).to_euler(), (0, 0, math.radians(10)))


class Chains(unittest.TestCase):

    def test_a_lamp_of_three_joints(self):
        s3d.empty_scene()
        base = s3d.cylinder("base", 100, 20)
        lower = s3d.at(s3d.box("lower", 10, 10, 200), z=20)
        upper = s3d.at(s3d.box("upper", 10, 10, 150), z=220)
        head = s3d.at(s3d.box("head", 40, 10, 10), x=20, z=370)
        s3d.joint("shoulder", lower, base.face("top"), axis="+Y", range=(-90, 90))
        s3d.joint("elbow", upper, lower.face("top"), axis="+Y", range=(-120, 120))
        s3d.joint("wrist", head, upper.face("top"), axis="+Y", range=(-90, 90))
        lamp = s3d.assembly("lamp", base, lower, upper, head)
        s3d.pose(lamp, 90, joint="elbow")
        close(self, at(head), (150 + 0, 0, 220 - 20))
        s3d.pose(lamp, -90, joint="shoulder")
        close(self, at(upper), (-200, 0, 20))
        close(self, at(head), (-180, 0, 170))              # the elbow still bent: its tip turned up


class Errors(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.cab = s3d.box("cab", 600, 600, 850)
        self.door = s3d.at(s3d.box("door", 20, 560, 800), x=310, z=25)
        s3d.joint("hinge", self.door, s3d.bbox(self.cab, "+X-Y"), axis="-Z", range=(0, 143))

    def test_out_of_range(self):
        for value, shown in ((143.5, "143.5°"), (-1, "-1°")):
            with self.assertRaises(ValueError) as caught:
                s3d.pose(self.door, value)
            self.assertEqual(str(caught.exception), f"pose 'hinge': {shown} is outside its range (0, 143)")
        drawer = s3d.at(s3d.box("drawer", 40, 150, 60), x=290, z=780)
        s3d.joint("slide", drawer, s3d.bbox(self.cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
        with self.assertRaisesRegex(ValueError, r"^pose 'slide': 201 is outside its range \(0, 200\)$"):
            s3d.pose(drawer, 201)
        s3d.pose(self.door, 143)
        s3d.pose(drawer, 200)
        close(self, at(drawer), (490, 0, 780))

    def test_a_free_joint_has_no_value(self):
        plug = s3d.at(s3d.box("plug", 10, 10, 10), x=-400)
        s3d.joint("cord", plug, self.cab.face("-X"), kind="free")
        with self.assertRaisesRegex(ValueError, r"^pose 'cord': a free joint has no value to pose: it moves every "
                                                r"way$"):
            s3d.pose(plug, 1)

    def test_a_joint_name_missing_unknown_or_shared(self):
        root = s3d.group("machine", self.cab, self.door)
        with self.assertRaisesRegex(ValueError, r"^pose: no joint named 'hing' on 'machine' or under it: the joints "
                                                r"under it are 'hinge'$"):
            s3d.pose(root, 1, joint="hing")
        with self.assertRaisesRegex(ValueError, r"^pose: 'cab' has no joint: s3d.joint\(...\) says how a part moves$"):
            s3d.pose(self.cab, 1)
        twin = s3d.transform(s3d.copy(self.door), mirror="y")
        twin.parent = root
        with self.assertRaisesRegex(ValueError, r"^pose: 'hinge' names 2 joints under 'machine' \(on 'door': copies "
                                                r"carry their joints\): pose the part itself, s3d.pose\(part, value\)$"):
            s3d.pose(root, 1, joint="hinge")
        s3d.pose(twin, 10)
        with self.assertRaisesRegex(TypeError, r"^pose 'hinge': value is a number, not '10'$"):
            s3d.pose(self.door, "10")
        with self.assertRaisesRegex(TypeError, r"^pose: ob is a jointed part, or a group above it with joint=, not "
                                               r"Frame$"):
            s3d.pose(s3d.Frame(), 1)

    def test_a_part_placed_on_a_moving_part_outside_its_group(self):
        sticker = s3d.at(s3d.box("sticker", 50, 5, 50), on=self.door.face("+X"))
        with self.assertRaisesRegex(ValueError, r"^pose: 'sticker' was placed on a frame read on 'door', which this "
                                                r"pose moves, and it is in no group that moves with it: it would stay "
                                                r"behind — group it with 'door', or give it a joint of its own$"):
            s3d.pose(self.door, 10)
        self.assertEqual(joints_of(self.door)[0]["value"], 0.0)            # nothing moved
        s3d.join(self.door, sticker)                                        # in the door now
        s3d.pose(self.door, 10)
        s3d.pose(self.door, 0)
        badge = s3d.extrude("badge", s3d.circle(20), 2, on=self.door.face("+X"))
        with self.assertRaisesRegex(ValueError, r"^pose: 'badge' was placed on a frame read on 'door'"):
            s3d.pose(self.door, 20)
        s3d.at(badge, on=s3d.Frame((330, 0, 400), (1, 0, 0)))               # placed elsewhere: it may stay
        s3d.pose(self.door, 20)

    def test_a_frame_read_on_a_part_used_up_since(self):
        boss = s3d.at(s3d.cylinder("boss", 30, 20), z=850)
        top = boss.face("+Z")
        s3d.join(self.cab, boss)
        cap = s3d.at(s3d.cylinder("cap", 20, 10), on=top)           # placed as on a frame built by hand
        self.assertNotIn("s3d~on", cap)
        close(self, at(cap), (0, 0, 870))

    def test_a_joint_declared_on_a_mechanism_not_at_rest(self):
        s3d.pose(self.door, 10)
        knob = s3d.at(s3d.box("knob", 10, 10, 10), x=330, z=400)
        message = (r"^joint 'k': the mechanism is not at rest — 'hinge' on 'door' is posed at 10: declare joints at "
                   r"rest, every joint at 0 \(pose them back to 0, or declare the joints first\)$")
        with self.assertRaisesRegex(ValueError, message):
            s3d.joint("k", knob, s3d.bbox(self.door, "+X"), kind="continuous")      # its parent is posed
        with self.assertRaisesRegex(ValueError, message.replace("'k'", "'slide'")):
            s3d.joint("slide", self.door, s3d.bbox(self.cab, "+X"), kind="prismatic", axis="+Z", range=(0, 1))
        s3d.joint("lid", s3d.at(s3d.box("lid", 10, 10, 10), z=860), self.cab.face("top"), range=(0, 90))
        with self.assertRaisesRegex(ValueError, message.replace("'k'", "'gear'")):     # what it follows is posed
            s3d.joint("gear", s3d.at(s3d.box("gear", 10, 10, 10), y=400), self.cab.face("top"),
                      kind="continuous", follows=self.door)
        s3d.pose(self.door, 0)                                                         # another chain may rest
        s3d.joint("k", knob, s3d.bbox(self.door, "+X"), kind="continuous")

if __name__ == "__main__":
    unittest.main()
