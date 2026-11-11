"""Construction labels, in program units, independent of later placement."""
NAMES = ('.s3d_s', '.s3d_x', '.s3d_y', '.s3d_extent')
def height_coordinates(ob, height):
    layers = [ob.data.attributes.new(name=n, type='FLOAT', domain='POINT') for n in NAMES]
    for v in ob.data.vertices:
        for a,value in zip(layers,(v.co.z,v.co.x,v.co.y,height)):
            a.data[v.index].value = value

def face_u(mesh):
    s,e = mesh.attributes.get(NAMES[0]),mesh.attributes.get(NAMES[-1])
    if s is None or e is None:
        return None
    return [sum(s.data[i].value/e.data[i].value if e.data[i].value else float('nan') for i in f.vertices)/len(f.vertices)
            for f in mesh.polygons]
