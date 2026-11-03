import sand3d as s3d


def rack(n=4, pitch=30, depth=40, t=5):
    base = s3d.extrude("base", s3d.rect(n * pitch, depth, r=3), t)
    pegs = s3d.at(s3d.cylinder("peg", 6, 20), on=s3d.grid(n, 1, pitch), z=t)         # a copy per frame
    hook = s3d.at(s3d.extrude("hook", s3d.rect(4, 12), 12), x=n * pitch / 2 - 4, z=t)
    image = s3d.transform(s3d.copy(hook), mirror="x")                                # its image across x = 0
    s3d.join(base, pegs, hook, image)                                                # a list is its elements
    s3d.hole(base, 4, on=s3d.grid(2, 1, (n - 1) * pitch).move(y=depth / 2 - 6))
    s3d.chamfer(base, 0.6, edges="peg")
    return base
