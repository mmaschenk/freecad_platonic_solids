import FreeCAD
from math import sqrt
import numpy as np

PHI = (1 + sqrt(5))/2
INVPHI = 1/PHI
KSI = sqrt((5-sqrt(5))/2)

triangular = 180

platonic_solids = {
    'tetrahedron': {
        'vertices' : [ (1,1,1), (1, -1, -1,), (-1, 1, -1), (-1, -1, 1) ],
        'faces': [ (0,1,2), (0,1,3), (0,3,2), (1,2,3)],
        'edges': [ (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) ],
        'radius': sqrt(3),
        'edgelength': 2*sqrt(2), 
    },
    'cube': {
        'vertices': [ (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                      (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
        'faces': [ (0,1,2,3), (1,2,6,5), (0,1,5,4), (0,3,7,4), (3,2,6,7), (4,5,6,7)],
        'edges': [ (0,1), (0,3), (0,3), (0,4), (1,2), (1,5), (2,3), (2,6),
                   (3,7), (4,5), (4,7), (5,6), (6,7)],
        'radius': sqrt(3),
        'edgelength': 2,
    },
    'octahedron': {
        'vertices': [ (0, 0, -1), (-1,0,0), (0,-1,0), (1,0,0), (0,1,0), (0,0,1)],
        'faces': [ (0,1,2), (0,2,3), (0,3,4), (0,4,1), (5,1,2), (5,2,3), (5,3,4), (5,4,1)],
        'radius': 1,
        'edgelength': sqrt(2),
    },
    'icosahedron': {
        'vertices': [ (0, 1, -PHI), (0, -1, -PHI), (-PHI, 0, -1), (PHI,0, -1), 
                      (-1,-PHI,0), (1, -PHI,0), (1, PHI, 0), (-1,PHI,0),
                      (-PHI,0, 1), (PHI,0,1), (0,1,PHI), (0,-1,PHI), ],

        'faces': [ (0,1,2), (0,1,3), (1,2,4), (1,4,5), (1,5,3), (3,5,9), (3,9,6),
                   (3,6,0), (0,6,7), (0,7,2), (2,7,8), (2,8,4), (4,8,11), (4,11,5),
                   (5,11,9), (9,10,6), (6,10,7), (7,10,8), (8,10,11), (11,10,9), ],
        'radius': KSI * PHI,
        'edgelength': 2,
    },
    'dodecahedron': {
        'vertices': [ (-INVPHI, 0, -PHI), (INVPHI, 0, -PHI), 
                      (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), 
                      (0, -PHI, -INVPHI), (0, PHI, -INVPHI),
                      (-PHI,-INVPHI,0), (PHI, -INVPHI,0), (PHI,INVPHI,0),(-PHI,INVPHI,0),
                      (0, -PHI, INVPHI), (0, PHI, INVPHI),
                      (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1), 
                      (-INVPHI, 0, PHI), (INVPHI, 0, PHI), ],
        'faces': [ (0,1,3,6,2), (0,1,4,7,5), (0,2,8,11,5), (1,3,9,10,4),
                   (2,6,12,14,8), (3,9,15,12,6), (4, 7,13,16,10), (5,11,17,13,7),
                   (8,14,18, 17,11), (10,16,19,15,9), (13,17,18,19,16), (12,15,19,18,14)],
        'radius': sqrt(3),
        'edgelength': 2/PHI,
    }
}


def clearobjects(doc):
    for obj in doc.Objects:
        #print(obj)
        doc.removeObject(obj.Name)

def normalize( tup, factor):
    return tuple( [x * factor for x in tup])

def vnormalize( tup, factor=1):
    return FreeCAD.Vector( tup[0]*factor, tup[1]*factor, tup[2]*factor)

def perpendicularcircle(wire,radius):
    start = wire.Edges[0].firstVertex().Point
    direction = wire.Edges[0].tangentAt(0)
    c = Part.makeCircle(radius, start, direction)
    return c

def solidify(doc, polygon, name='sweep', radius=0.1):
    wire = polygon.Shape.Wires[0]
    c = perpendicularcircle(wire, radius)
    circleObj = doc.addObject("Part::Feature", f"{name}-outline")
    circleObj.Shape = c
    doc.recompute()
    #Part.show(c)
    #solid = wire.makePipe(c)
    #Part.show(solid)

    sw = doc.addObject('Part::Sweep', name)
    sw.Sections = circleObj
    sw.Spine = (polygon,[])
    sw.Solid = True
    sw.Frenet = True
    sw.Transition = 'Round corner'
    circleObj.Visibility = False
    polygon.Visibility = False
    doc.recompute()
    return sw
    
def solidwire(doc, wire, name="solidwire", radius=0.1):
    polygonObj = doc.addObject("Part::Feature", f"{name}-polygon")
    polygonObj.Shape = wire
    doc.recompute()
    return solidify(doc, polygonObj, name=name, radius=radius)    

def determinefactor(platonic_solid, outerradius=None, edgelength=None):
    radius = platonic_solid['radius']

    if outerradius:
        factor = outerradius / radius
        print(f"Outerradius: {outerradius}")
    elif edgelength:
        factor = edgelength / platonic_solid['edgelength']
    else:
        raise Exception('Neither outterradius or edgelenght specified. One of these must be present')
    return factor

def platonicsolid(solid='cube', name=None, clearall=False, vertexpolygon=False, cloud=False,
                  solidoutline=False, solidobject=True, outerradius=None, edgelength=None):
    platonic_solid=platonic_solids[solid]

    if name == None:
        name = solid

    doc = FreeCAD.activeDocument()

    if clearall:
        clearobjects(doc)

    #print(doc.supportedTypes())
    radius = platonic_solid['radius']

    if outerradius:
        factor = outerradius / radius
        print(f"Outerradius: {outerradius}")
    elif edgelength:
        factor = edgelength / platonic_solid['edgelength']

    print(f"Factor: {factor} = {factor/platonic_solid['radius']} * R")
    print(f"Edge length: {platonic_solid['edgelength']}")
    if vertexpolygon:
        p = Part.makePolygon( [ normalize(x, factor) for x in platonic_solid['vertices']])
        solidwire(doc, p, name=f"{name}_vertices", radius=factor*platonic_solid['edgelength']/20)
    
    platonic_solid['edges'] = []
    if cloud:
        spheres = []
        for i, vertex in enumerate(platonic_solid['vertices']):
            s = doc.addObject("Part::Sphere", f"{name}_sphere{i}")
            s.Radius = 0.2
            p = FreeCAD.Placement()
            p.Base = FreeCAD.Vector( normalize( (vertex[0], vertex[1], vertex[2]), factor))
            s.Placement = p
            s.Visibility = False
            spheres.append(s)

        spherefusion = doc.addObject("Part::MultiFuse", f"{name}_cloud")
        spherefusion.Shapes = spheres
    
    facelist = []
    solidfaces = []
    for i, face in enumerate(platonic_solid['faces']):
        edgelist = []
        verts = platonic_solid['vertices']
        prev = None
        for v in face:
            if prev is not None:
                edge = Part.makeLine( normalize( verts[prev], factor), normalize( verts[v], factor ))
                edgelist.append(edge)
            prev = v
        edge = Part.makeLine( normalize( verts[prev], factor), normalize( verts[face[0]], factor) )
        edgelist.append(edge)
        wire = Part.Wire(edgelist)
        if solidoutline:
            solidfaces.append(solidwire(doc, wire, f"{name}_face{i}", radius=factor*platonic_solid['edgelength']/20))
        face = Part.Face(wire)
        facelist.append(face)

    if solidoutline:
        solidfacefusion = doc.addObject("Part::MultiFuse", f"{name}_solidfaces")
        solidfacefusion.Shapes = solidfaces

    if solidobject:
        shell = Part.makeShell(facelist)
        solid = doc.addObject( "Part::Feature", name)
        solid.Shape = Part.makeSolid(shell)

def matricize(faces):
    def set(x,y):
        matrix[x][y] = True
        matrix[y][x] = True

    def unset(x,y):
        matrix[x][y] = False
        matrix[y][x] = False

    size = max([y for x in faces for y in x])+1
    matrix = np.zeros( (size,size), dtype=bool )

    for f in faces:
        lasti = None
        firsti = None
        for i in f:
            if lasti is not None:
                set(i,lasti)
            else:
                firsti = i
            lasti = i
        set(lasti,firsti)        
    return matrix

def simplepaths(faces):
    size = max([y for x in faces for y in x])+1
    matrix = np.zeros( (size,size), dtype=bool )

    def set(x,y):
        matrix[x][y] = True
        matrix[y][x] = True

    def unset(x,y):
        matrix[x][y] = False
        matrix[y][x] = False

    def getfirst(matrix):
        x, y = matrix.nonzero()
        unset(x[0],y[0])
        return x[0], y[0]

    for f in faces:
        lasti = None
        firsti = None
        for i in f:
            if lasti is not None:
                set(i,lasti)
            else:
                firsti = i
            lasti = i
        set(lasti,firsti)        

    result = []
    while (np.any(matrix)):
        x,y = getfirst(matrix)
        curResult = [x, y]
        result.append(curResult)
        #print(f"Starting: {x}\nOutputting: {y}")

        deadend = False
        while not deadend:
            x1,  = matrix[y].nonzero()
        
            if len(x1):
                y1 = x1[0]
                #print(f"Outputting: {y1}")
                curResult.append(y1)
                unset(y,y1)
                y = y1
            else:
                deadend = True
    print(result)
    return result

def cylindershapes(solid, factor=1, radius=0.2):
    p = platonic_solids[solid]
    faces = p['faces']
    verts = p['vertices']
    matrix = matricize(faces)

    shapes = []
    xrange,yrange = matrix.shape
    
    for x in range(xrange):
        for y in range(x+1,yrange):
            if matrix[x][y]:
                start = vnormalize(verts[x],factor)
                end = vnormalize(verts[y], factor)
                direction = end-start
                s = Part.makeCylinder(radius, direction.Length, start, direction)
                shapes.append(s)
    return shapes

def sphereshapes(solid, factor=1, radius=0.2):
    p = platonic_solids[solid]
    verts = p['vertices']

    shapes = []
    for v in verts:
        center = vnormalize(v, factor)
        s = Part.makeSphere(radius, center)
        shapes.append(s)
    return shapes

def cylinders(solid, outerradius=None, edgelength=None, edgeradius=None):
    platonic_solid = platonic_solids[solid]

    factor = determinefactor(platonic_solid, outerradius=outerradius, edgelength=edgelength)
    print(f"Factor: {factor}")

    if edgeradius is None:
        edgeradius = factor*platonic_solid['edgelength']/20
        print(f"Edgeradius redetermined at {edgeradius}")

    cylinders = cylindershapes(solid, factor=factor, radius=edgeradius)
    spheres = sphereshapes(solid, factor=factor, radius=edgeradius)

    doc = FreeCAD.activeDocument()
    clearobjects(doc)
    objects = []

    for s in cylinders:
        cyl = doc.addObject("Part::Feature", "cylinder")
        cyl.Shape = s
        objects.append(cyl)

    for s in spheres:
        cyl = doc.addObject("Part::Feature", "sphere")
        cyl.Shape = s
        objects.append(cyl)

    spherefusion = doc.addObject("Part::MultiFuse", "fuse")
    spherefusion.Shapes = objects

def looseedges(solid):
    p = platonic_solids[solid]
    faces = p['faces']
    verts = p['vertices']
    matrix = matricize(faces)

    doc = FreeCAD.activeDocument()
    #clearobjects(doc)

    print(matrix)
    xrange,yrange = matrix.shape
    
    first = False
    for x in range(xrange):
        for y in range(x+1,yrange):
            print(x, y, matrix[x][y])
            if matrix[x][y]:
                edge = Part.makeLine( verts[x], verts[y] )
                wire = Part.Wire(edge)
                if first:                    
                    s = solidwire(doc, wire, f"name" )  
                    body = doc.addObject("PartDesign::Body", "Body")
                    body.BaseFeature = s
                    doc.recompute()
                    s.Visibility = False      
                    first = False
                else:
                    #s = Part.makeCylinder(0.2, 5, FreeCAD.Vector(0,0,0), FreeCAD.Vector(1,1,0))
                    start = vnormalize(verts[x])
                    end = vnormalize(verts[y])
                    direction = end-start
                    print(f"Start: {start}")
                    print(f"Direction: {direction}")
                    s = Part.makeCylinder(0.2, direction.Length, start, direction)
                    #cyl = doc.addObject("Part::Cylinder", "cylinder")
                    #cyl.Radius = 0.2
                    #cyl.Shape = s
                    cyl = doc.addObject("Part::Feature", "cylinder")
                    cyl.Shape = s
                    #Part.show(cyl)
                    

    
def paths(solid):
    p = platonic_solids[solid]
    verts = p['vertices']
    sp = simplepaths(p['faces'])

    doc = FreeCAD.activeDocument()
    clearobjects(doc)
    wirelist = []
    for n, w in enumerate(sp):
        prevv = None
        edgelist = []
        for v in w:
            print('v:',v)
            if prevv is not None:
                edge = Part.makeLine( verts[prevv], verts[v] )
                edgelist.append(edge)
            prevv = v
        print(edgelist)
        wire = Part.Wire(edgelist)
        wirelist.append(wire)
        #Part.show(wire)
        if n == 0:
            s = solidwire(doc, wire, f"name" )        
            body = doc.addObject("PartDesign::Body", "Body")
            body.BaseFeature = s
            doc.recompute()
            s.Visibility = False
        else:
            s = solidwire(doc, wire, f"name" )
            print(s.Name)
            sec = s.Sections[0]
            print(sec.Name)
            (spine,_) = s.Spine
            print(spine.Name)
            doc.removeObject(s.Name)
            sketch = Draft.makeSketch(sec,autoconstraints=True)
            print(sketch)
            profile = Draft.makeSketch(spine,autoconstraints=True)
            #body.Group.append(sketch)
            body.addObject(sketch)
            #body.addObject(spine)
            #p = body.newObject('PartDesign::AdditivePipe','AdditivePipe')
            print(f'Added {p}')
            #p.Profile = sketch
            #doc.recompute()
            print('group',body.Group)
            sb = doc.addObject("PartDesign::ShapeBinder", "shapebinder")
            sb.Shape = spine.Shape
            body.addObject(sb)

    #aw = Part.Wire(wirelist)
    #Part.show(aw)
        
    #solidwire(doc, aw, f"name" )
    

def main():
    solid='cube'
    #solid='tetrahedron'
    #solid='octahedron'
    #solid='icosahedron'
    solid='dodecahedron'
    
    
    #platonicsolid(solid,  outerradius=sqrt(3/2), clearall=True, solidobject=True, solidoutline=False, cloud=True, vertexpolygon=False)
    #platonicsolid(solid,  outerradius=15, clearall=True, solidobject=True, solidoutline=False, cloud=False, vertexpolygon=True)
    #platonicsolid(solid, edgelength=2, clearall=True, solidobject=True, solidoutline=False, cloud=False, vertexpolygon=False )
    #simplepaths(platonic_solids[solid]['faces'])
    #paths(solid)
    cylinders(solid, edgelength=20, edgeradius=2)

if __name__ == "__main__":
    main()
    #newobj = doc.addObject("Part::Feature", "newPart")
