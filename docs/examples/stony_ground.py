"""Stones and fallen leaves strewn on uneven ground, off its steep slopes and out of its dry channel."""
import math

import numpy as np
import sand3d as s3d


def stony_ground(size=1200, depth=150, relief=140, channel=150, steepest=20, stones=24, leaves=30,
                 stone=110, leaf=50, seed=0):
    ground = ground_part(size, depth, relief, channel, seed)
    sites = open_ground(ground, stones + leaves, size / 2 - stone, channel, steepest, seed)
    sizes = s3d.vary(stone, stone / 3, stones, seed=seed)
    turns = s3d.vary(0, 180, leaves, seed=seed)
    rocks = [s3d.at(stone_part(f"stone{k}", d, seed + k), on=site.offset(-d / 4))     # sunk a quarter
             for k, (site, d) in enumerate(zip(sites[:stones], sizes))]
    litter = [s3d.lay(s3d.at(leaf_part(f"leaf{k}", leaf), on=site.turn(rz=turn)), ground)
              for k, (site, turn) in enumerate(zip(sites[stones:], turns))]
    post = s3d.material(s3d.cylinder("post", stone / 3, depth * 2), "pine")
    foot = s3d.snap(ground, [(0, -size / 3, depth * 4)], direction="-Z")[0]
    marker = s3d.at(post, on=s3d.Frame(foot.origin))          # upright, not along the slope
    return s3d.assembly("stony_ground", ground, rocks, litter, marker)


def ground_part(size, depth, relief, channel, seed):
    """A slab whose top rolls, with a dry channel running along X."""
    def rise(x, y, z):
        hills = s3d.noise(x * 3 / size, y * 3 / size, z * 3 / size, seed=seed)
        return np.clip(hills - np.exp(-(y / channel) ** 2), -1, 1)
    ground = s3d.box("ground", size, size, depth)
    s3d.displace(ground, relief, channel / 2, noise=rise, direction="+Z", faces="top")
    return s3d.material(ground, "soil")


def open_ground(ground, n, reach, channel, steepest, seed):
    """Sites on the ground's top, within reach of its centre, off steep slopes and out of the channel."""
    sites = s3d.scatter(10 * n, ground, seed=seed)       # the top is a fraction of the surface
    level = math.cos(math.radians(steepest))
    return s3d.Frames([f for f in sites if f.normal.z > level and abs(f.origin.y) > channel
                       and max(abs(f.origin.x), abs(f.origin.y)) < reach])[:n]


def stone_part(name, d, seed):
    points = s3d.scatter(14, (d, d * .8, d * .6), seed=seed, surface=True)
    return s3d.material(s3d.hull(name, points.jitter(x=d / 8, y=d / 8, z=d / 8, seed=seed)), "rock")


def leaf_part(name, length):
    return s3d.material(s3d.extrude(name, s3d.ellipse(length, length / 2), length / 30), "leaf")
