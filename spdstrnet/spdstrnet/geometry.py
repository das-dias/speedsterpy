"""_summary_
geometry.py contains the main algorithms
for the preprocessing of layout geometry
in order to enable point to point resistance extraction

[author]    Diogo André Silvares Dias
[date]      2022-04-17
[contact]   das.dias@campus.fct.unl.pt
"""
from loguru import logger
import itertools
from enum import Enum
import warnings
from operator import contains
import numpy as np
from gdstk import(
    Library,
    Cell,
    Polygon,
    rectangle,
    RobustPath,
    Reference,
    # functions
    boolean, # perform boolean operations on two polygon sets
    inside,
    slice
)
from .data import(
    SpeedsterPort,
    SpeedsterPortType,
)
from spdstrutil import (
    GdsTable,
    GdsLayerPurpose,
    timer,
)
from copy import copy, deepcopy


class PolygonDirection(Enum):
    """_summary_
    Enum class to define the direction of 
    a polygon in relation to its neighbour
    Direction:
        0: no direction
        1: north
        2: north-east
        3: east
        4: south-east
        5: south
        6: south-west
        7: west
        8: north-west
    """
    
    None
    NORTH       = (0,1)#"+y"
    NORTH_EAST  = (1,1)
    EAST        = (1,0)#"+x"
    SOUTH_EAST  = (1,-1) 
    SOUTH       = (0,-1)#"-y"
    SOUTH_WEST  = (-1,-1)
    WEST        = (-1,0)#"-x"
    NORTH_WEST  = (-1,1)

def check_point_inside_polygon(
    polygon,
    point: tuple,
) -> bool:
    """_summary_
    Checks if a point is inside a polygon
    Args:
        polygon (Polygon): Polygon Set object
        point   (list)      : [x:float, y: float] items list
    Returns:
        bool: returns wether the point is inside the polygon or not
    """
    if type(point[0]) != float or len(point) != 2:
        raise ValueError("Point must be a tuple of 2 floats.")
    return inside([point], polygon)[0]

def check_polygon_overlap(
    polygonA,
    polygonB,
    layer: int = 0,
    datatype: int = 0,
    precision = 1e-3,
) -> list:
    """_summary_
    Checks if two polygons overlap, and 
    returns the overlapping region, in case they do
    Args:
        polygonA (Polygon): Polygon object
        polygonB (Polygon): Polygon object
        layer    (int)       : layer of the resulting polygon from the boolean operation
        dataType (int)       : datatype of the resulting polygon from the boolean operation
        precision(float)     : precision of the cuts
    """
    polyList = boolean(
        polygonA, 
        polygonB,
        'and', 
        layer = layer, 
        datatype = datatype, 
        precision = precision
    ) 
    return polyList if len(polyList) > 0 else None

def get_common_edges(
    polyA,
    polyB
) -> list:
    pointsA = polyA.points
    pointsB = polyB.points
    edges = []
    def _superposed_collinear_edge(e1, e2):
        """_summary_
        Checks if two edges e1 and e2 are collinear and intersect
        to infer about their intersection, returning it in case 
        it exists
        Args:
            e1 ( array( [p1, p2] ) ): edge1/vector1
            e2 ( array( [p1, p2] ) ): edge2/vector2

        Returns:
            ( array( [p1, p2] ) ) : the superposed edge
        """
        # compute colinearity
        # get 3D vectors defined by e1 and e2
        v1 = [ e1[1][0] - e1[0][0], e1[1][1]- e1[0][1], 0 ]
        v2 = [ e2[1][0] - e2[0][0], e2[1][1]- e2[0][1], 0 ]
        # get cross product
        cp = np.cross(v1,v2)
        norm_cp = np.sqrt(cp.dot(cp))
        # infer colinearity
        if norm_cp != 0.0: # colinearity exists when |v1xv2| = 0
            return None
        # compute intersection
        # get the straight line defined by e1 : y = ax + b -> y-y0 = a * (x-x0) where a = (y1-y0)/(x1-x0) and b = y0 - a*x0
        with warnings.catch_warnings():
            warnings.simplefilter("error") # set warnings to error to catch div by zero
            try:
                a = (e1[1][1] - e1[0][1]) / ( (e1[1][0] - e1[0][0]) )
                b = e1[0][1] - a*e1[0][0]
                # check if the first point of e2 is on top of the straight line defined by the vertices of e1
                if not np.equal(e2[0][1], ( a*e2[0][0] + b ) ): # y2 == a*x2 + b ?
                    return None
            except Exception: # division by zero, infinite slope, the defined line is : x = e1[0][0]
                a = None
                # only have to check if the abciss of e2[0] == e1[0][0]
                if not np.equal(e2[0][0], e1[0][0]):
                    return None
        # if it is, check if either the final or the sstarting point of e2 is in the bounding box defined by the vertices of e1
        bbox = rectangle( e1[0], e1[1] )
        bbox2 = rectangle( e2[0], e2[1] )
        if not( inside([e2[0]], bbox)[0] or inside([e2[1]], bbox)[0] ):
            if not ( inside([e1[0]], bbox2)[0] or inside([e1[1]], bbox2)[0] ):
                return None
        # finally, there is actually an intersection, and we must return it
        # check e2 fully contained in e1
        if inside([e2[0]], bbox)[0] and inside([e2[1]], bbox)[0]:
            return e2
        # check e1 fully contained in e2
        if inside([e1[0]], bbox2)[0] and inside([e1[1]], bbox2)[0]:
            return e1
        # check for partial intersection
        # todo : Not perfect! Polygons sharing a single vertix are still considered as cheighbours
        if inside([e2[0]], bbox)[0]:
            return (e2[0], e1[1])
        
        if inside([e2[1]], bbox)[0]:
            return (e1[0], e2[1])
        
        return None
        
    for edgeA in zip(pointsA, pointsA[1:]):
        for edgeB in zip(pointsB, pointsB[1:]):
            superposed = _superposed_collinear_edge(edgeA, edgeB)
            if superposed is not None:
                edges.append(list(superposed))
        # check last edge
        edgeB = (pointsB[-1], pointsB[0])
        superposed = _superposed_collinear_edge(edgeA, edgeB)
        if superposed is not None:
            edges.append(list(superposed))
    # LAST ITERATION 
    # check last edge
    edgeA = (pointsA[-1], pointsA[0])
    for edgeB in zip(pointsB[:-1], pointsB[1:]):
        superposed = _superposed_collinear_edge(edgeA, edgeB)
        if superposed is not None:
            edges.append(list(superposed))
    # check last edge
    edgeB = (pointsB[-1], pointsB[0])
    superposed = _superposed_collinear_edge(edgeA, edgeB)
    if superposed is not None:
        edges.append(list(superposed))
    # convert edges to list of tuples and filter the edges defining a single point
    edges = [ [tuple(p) for p in edge] for edge in edges if not np.array_equal(edge[0], edge[1])]
    return edges if len(edges)>0 else None


def bool_polygon_overlap_check(
    polygonA,
    polygonB,
    lock = False
) -> bool:
    """_summary_
    Performs a boolean check 
    to infer if the two polygons overlap
    Args:
        polygonA (Polygon): Polygon object
        polygonB (Polygon): Polygon object
    Returns:
        bool:   returns wether the two polygons overlap or not
                True: They overlap; False: They don't overlap
    """
    return not ( check_polygon_overlap(polygonA, polygonB) is None )

def check_neighbour_polygons(
    polyA,
    polyB,
) -> bool:
    """_summary_
    Checks if the two polygons have common edges and don't overlap!
    Args:
        polygonA (Polygon): Polygon object
        polygonB (Polygon): Polygon object
    Returns:
        bool:   returns wether the two polygons have common edges
                True: They have common edges; False: They don't have common edges
    """
    if polyA.layer == polyB.layer and polyA.datatype == polyB.datatype:
        if get_common_edges(polyA,polyB) is not None:
            return True if not bool_polygon_overlap_check(polyA,polyB) else False
    return False

def check_same_polygon(
    polyA,
    polyB,
    precision = 1e-3,
) -> bool:
    """_summary_
    Checks if two polygons are the same
    Args:
        polyA       (np.array)  : vertices of the polygon
        polyB       (np.array)  : vertices of the polygon
        precision   (float)     : precision of the comparison
    Returns:
        bool: returns wether the two polygons are the same or not
                True : They are the same; False: They are not the same
    """
    
    # check if the layers and datatype are the same
    if polyA.layer != polyB.layer:
        return False
    layer = polyA.layer
    if polyA.datatype != polyB.datatype:
        return False
    datatype = polyA.datatype
    # check if the interception of the two polygons is equal to both of them, or
    # if the not operation (polyA - polyB) and (polyB - polyA) is equal to an empty space of points/polygons
    notAB = boolean( polyA, polyB, "not" )
    notBA = boolean( polyB, polyA, "not" )
    return len(notAB) == 0 and len(notBA) == 0

def check_polygon_contains_polygon(
    polyA,
    polyB,
) -> bool:
    """_summary_
    Checks if polygon A contains polygon B
    Args:
        polyA       (np.array)  : vertices of the polygon
        polyB       (np.array)  : vertices of the polygon
        maxPoints   (int)       : maximum number of points inside the polygon
        precision   (float)     : precision of the comparison
    Returns:
        bool: returns wether polygon A contains polygon B or not
                True : polygon A contains polygon B; False: polygon A doesn't contain polygon B
    """
    # check if the layers and datatype are the same
    if polyA.layer != polyB.layer:
        return False
    if polyA.datatype != polyB.datatype:
        return False
    # if the union of the two polygons is equal to the polygon A, then polygon A contains polygon B
    # which is equal to checking if : (polyA U polyB) NOT polyA == 0
    unionAB = boolean(
        polyA, 
        polyB, 
        "or",
        layer = polyA.layer,
        datatype = polyA.datatype
    )
    boolMap = [check_same_polygon(uniAB, polyA) for uniAB in unionAB]
    return (False not in boolMap) or check_same_polygon(polyA, polyB)
    
def find_centroid(
    poly,
) -> list:
    """_summary_
    Finds the centroid of a single polygon
    Args:
        poly (np.array): vertices of the polygon
    Returns:
        list: [x:float, y: float] 2 items list
    """
    cent = [0,0]
    signedArea = 0.0
    points = poly.points
    nPoly = len(points)
    # for all the vertices of the polygon
    for k, point in enumerate(points):
        x0 = point[0]
        y0 = point[1]
        opposite_point = points[(k+1) % nPoly]
        x1 = opposite_point[0]
        y1 = opposite_point[1]
        # compute the value of the area composed of the two vertices
        # using the shoelace formula
        A = x0*y1 - x1*y0
        signedArea += A
        # compute the centroid's coordinates
        cent[0] += (x0 + x1)*A
        cent[1] += (y0 + y1)*A
    # compute the centroid's coordinates
    signedArea *= 0.5
    cent[0] = cent[0]/ (6.0*signedArea)
    cent[1] = cent[1]/ (6.0*signedArea)
    return tuple(cent)


def unit_vector(
    pointA: list,
    pointB: list,
) -> list:
    """_summary_
    Returns the unit vector of a point
    Args:
        point (list) : [x:float, y: float] 2 items list
    Returns:
        list: [x:float, y: float] 2 items list
    """
    vec = np.array([pointB[0] - pointA[0], pointB[1] - pointA[1]])
    return [ vec[0]/np.sqrt(vec.dot(vec)), vec[1]/np.sqrt(vec.dot(vec)) ]



def saturate_vector(
    vec: list
) -> list:
    """_summary_
    Saturates a vector to a unit vector
    Args:
        vec (list): [vx:float, vy: float] the vector to saturate
    Returns:
        list: [vx, vy] the saturated vector
    """
    return [int(np.round(vec[0])), int(np.round(vec[1]))]


def check_neighbour_direction(
    poly,
    neighbour,    
) -> PolygonDirection:
    """_summary_
    In case the polygon intercepts
    its neighbour, returns the direction
    in which it happens
    Args:
        poly        (np.array): vertices of the polygon
        neighbour   (np.array): vertices of the neighbour
    Returns:
        PolygonDirection: the direction in which the polygon intercepts
    """
    # check if the polygon and its neighbour intersect
    if not check_neighbour_polygons(poly, neighbour):
        # if they don't, return no direction
        return None
    # if they do,
    # compute the centers of each polygon, assuming both are squares and convex hulls
    center1 = find_centroid(poly)
    center2 = find_centroid(neighbour)
    # from the center point of the polygons,
    # check the direction of the neighbour , in relation to the polygon
    direction_vec = unit_vector(center1, center2)
    # saturate the direction in North, South, East, West
    direction = tuple(saturate_vector(direction_vec))
    #return the direction
    if direction == PolygonDirection.NORTH.value:
        return PolygonDirection.NORTH
    elif direction == PolygonDirection.SOUTH.value:
        return PolygonDirection.SOUTH
    elif direction == PolygonDirection.EAST.value:
        return PolygonDirection.EAST
    elif direction == PolygonDirection.WEST.value:
        return PolygonDirection.WEST
    elif direction == PolygonDirection.NORTH_EAST.value:
        return PolygonDirection.NORTH_EAST
    elif direction == PolygonDirection.NORTH_WEST.value:
        return PolygonDirection.NORTH_WEST
    elif direction == PolygonDirection.SOUTH_EAST.value:
        return PolygonDirection.SOUTH_EAST
    elif direction == PolygonDirection.SOUTH_WEST.value:
        return PolygonDirection.SOUTH_WEST
    else: 
        return None


def get_direction_between_rects(
    poly,
    neighbour,    
) -> PolygonDirection:
    """_summary_
    Gets the direction between two rectangles, saturating this direction
    in only 4 directions : North, South, East, West
    # ! Deprecated
    Args:
        poly        (np.array): from rectangle
        neighbour   (np.array): to rectangle
    Returns:
        PolygonDirection: direction between the rectangles
    """
    raise DeprecationWarning("get_direction_between_rects is deprecated")
    if len(poly) != 4 or len(neighbour) != 4:
        raise ValueError("rectangles must have 4 vertices only!")
    # get the 8-side direction between the two rectangles
    direction = check_neighbour_direction(poly, neighbour)
    # check which side (horizontal or vertical) is the bigger
    verd = np.max( [y for y in poly[:,1]] ) - np.min( [y for y in poly[:,1]] )
    hord = np.max( [x for x in poly[:,0]] ) - np.min( [x for x in poly[:,0]] )
    retDir = direction
    if verd > hord:
        # if the vertical side is bigger,
        if direction in [PolygonDirection.NORTH_EAST, PolygonDirection.NORTH_WEST]:
            retDir = PolygonDirection.NORTH
        elif direction in [PolygonDirection.SOUTH_EAST, PolygonDirection.SOUTH_WEST]:
            retDir = PolygonDirection.SOUTH
        else:
            pass # proceed
    else:
        # if the horizontal side is bigger,
        if direction in [PolygonDirection.NORTH_EAST, PolygonDirection.SOUTH_EAST]:
            retDir = PolygonDirection.EAST
        elif direction in [PolygonDirection.SOUTH_WEST, PolygonDirection.NORTH_WEST]:
            retDir = PolygonDirection.WEST
        else:
            pass # proceed
    return retDir


def get_rectangle_length_width_ratio(
    poly,
    polyNeighbourDirection: PolygonDirection
) -> tuple:
    """_summary_
    Gets the ratio of Length/Width of the rectangle
    in relation of the direction of movement of a point
    through the surface of the rectangle
    Args:
        poly (gdstk.Polygon) : 
            Polygon to check the Length/Width ratio
            
        polyNeighbourDirection (PolygonDirection) : 
            The direction of the flow of a point charge through the surface of the polygon
    Returns:
        tuple: Length/Width ratio
    """
    if len(poly.points) != 4:
        raise ValueError("Polygon is not a rectangle")
    chargeMovementVec = np.array(polyNeighbourDirection.value)
    points = poly.points
    firstEdge = (points[0], points[1])
    vec1 = np.array([firstEdge[1][0] - firstEdge[0][0], firstEdge[1][1] - firstEdge[0][1]])
    secondEdge = (points[1], points[2])
    vec2 = np.array([secondEdge[1][0] - secondEdge[0][0], secondEdge[1][1] - secondEdge[0][1]])
    # compute colinearity for the first and second edge, in relation to the charge
    # movemenet vec
    # get cross product
    cpVec1ChargeMovementVec = np.cross(vec1, chargeMovementVec)
    # get cross product vector euclidean norm to measure collinearity
    collinearityVec1ChargeMovementVec = np.sqrt(cpVec1ChargeMovementVec.dot(cpVec1ChargeMovementVec))
    cpVec2ChargeMovementVec = np.cross(vec2, chargeMovementVec)
    collinearityVec2ChargeMovementVec = np.sqrt(cpVec2ChargeMovementVec.dot(cpVec2ChargeMovementVec))
    if collinearityVec1ChargeMovementVec > collinearityVec2ChargeMovementVec:
        #  secondEdge defines the length of the rectangle
        return np.sqrt( vec2.dot(vec2) ) / np.sqrt( vec1.dot(vec1) )
    else:
        return np.sqrt( vec1.dot(vec1) ) / np.sqrt( vec2.dot(vec2) )
        

def fragment_polygon(
    poly: Polygon,
    precision = 1e-3,
) -> list:
    """_summary_
    Fragments the polygon into multiple rectangular polygons
    Args:
        poly        (Polygon)   : the polygon to fragment
        maxPoints   (int)       : the maximum number of points to keep
        precision   (float)     : the precision to use
    Returns:
        Polygon: the polygon set resulting from the fragmentation operation
    NOTE: Optimal number of max points is 5, to obtain only rectangles in the
    fragmented polygon 
    """
    maxPoints = 5 # empirical maximum number of points to fractue everything in rectangles
    frags = poly.fracture(max_points = maxPoints, precision = precision)
    #ey = [0,1,0]
    ex = [1.0 , 0.0, 0.0]
    slicedFrags = []
    for p in frags:
        edge1 = (p.points[0], p.points[1])
        vector1 = np.array([edge1[1][0] - edge1[0][0], edge1[1][1] - edge1[0][1], 0])
        lengthEdge1 = np.sqrt(vector1.dot(vector1))
        edge2 = (p.points[1], p.points[2])
        vector2 = np.array([edge2[1][0] - edge2[0][0], edge2[1][1] - edge2[0][1], 0])
        lengthEdge2 = np.sqrt(vector2.dot(vector2))
        # check the colinearity of each of the edges with unit abciss vector
        cpVector1Ex = np.cross(ex,vector1)
        colVector1Ex = np.sqrt(cpVector1Ex.dot(cpVector1Ex))
        cpVector2Ex = np.cross(ex,vector2)
        colVector2Ex = np.sqrt(cpVector2Ex.dot(cpVector2Ex))
        slices = [p]
        cutPos = None
        if colVector1Ex > colVector2Ex:
            # compute the slicing points for the polygon along the edge 2 orientation
            if lengthEdge1 != 0.0 and lengthEdge2 > 1.0*lengthEdge1:
                cutPos = list( np.sort( np.linspace(edge2[0][0], edge2[1][0], num = int(lengthEdge2/lengthEdge1)+1)[1:-1]) )
                slices = slice( p, cutPos, "x",precision = precision)
            elif lengthEdge2 != 0.0 and lengthEdge1 > 1.0*lengthEdge2:
                cutPos = list( np.sort( np.linspace(edge1[0][1], edge1[1][1], num = int(lengthEdge1/lengthEdge2)+1)[1:-1] ) )
                slices = slice( p, cutPos, "y",precision = precision)
        else: # colVector1Ex <= colVector2Ex
            if lengthEdge1 != 0.0 and lengthEdge2 > 1.0*lengthEdge1:
                cutPos = list( np.sort( np.linspace(edge2[0][1], edge2[1][1], num = int(lengthEdge2/lengthEdge1)+1)[1:-1]) )
                slices = slice( p, cutPos, "y",precision = precision)
            elif lengthEdge2 != 0.0 and lengthEdge1 > 1.0*lengthEdge2:
                cutPos = list( np.sort( np.linspace(edge1[0][0], edge1[1][0], num = int(lengthEdge1/lengthEdge2)+1)[1:-1]) )
                slices = slice( p, cutPos, "x",precision = precision)
        [[slicedFrags.append(sl) for sl in subslice] for subslice in slices]
    return slicedFrags if len(slicedFrags) > 0 else None
        
    # determine the orientation of slicing
    # after fracturing, perform slicing on each of the resulting polygons
    # and finally retrieve the resulting sliced polygons
@timer
def fragment_net(
    name: str,
    cell: Cell,
    table: GdsTable,
    precision = 1e-3,
) -> Cell:
    """_summary_
    Fragments the net into a Cell of Polygon
    objects resulting from horizontal and vertical
    cuts in each polygon, and returns it
    Args:
        name        (str)       : the name of the cell
        cell        (Cell)   : Cell object containing the net
        maxPoints   (int)       : maximum number of points inside the polygon
        precision   (float)     : precision of the cuts
    """
    # create a new gds cell, to which the fragments will be added
    net = Cell(name)
    metalLayers = table.getDrawingMetalLayersMap()
    # filter metal layers, ignoring fragmentation of vias
    metals = [val for key,val in metalLayers.items() if contains(key, "met") ]
    for poly in itertools.chain(cell.get_polygons(), cell.get_paths()):
        if (poly.layer, poly.datatype) in metals:
            fragments = fragment_polygon(poly, precision = precision)
            if fragments is not None:
                [net.add(p) for p in fragments]
        else:
            net.add(poly)
    return net

def get_polygons_by_spec(
    cell: Cell,
    filters: list
) -> list:
    """_summary_
    Returns a list of polygons of a cell
    that features a layer and datatype tuple 
    that corresponds to a tuple featured in the
    filters list
    Args:
        cell    (Cell)  : GDS cell structure holding the polygons to be filtered
        filters (list)  : List of (layer,datatype) tuples that filter the polygons of the cell
    Returns:
        list: list of Polygons resulting from the filtering operation
    """
    if not all( [ type(filter) == tuple for filter in filters] ) and not all( [ type(filter) == list for filter in filters] ) and not all( [ len(filter) == 2 for filter in filters] ):
        raise TypeError("Each filter must be a list of tuples or lists of length 2!")
    polys = []
    for poly in itertools.chain(cell.get_polygons(), cell.get_paths(), cell.get_labels()):
        if (poly.layer, poly.datatype) in filters:
            polys.append(poly)
    return polys

def get_polygon_dict(
    cell: Cell,
    filters: list,
) -> dict:
    """_summary_
    Gets a dictionary of polygons by spec
    Args:
        cell        (Cell)   : Cell object containing the net
        spec        (list)      : list of layer, datatype and name
    Returns:
        dict: dictionary of {(layer,datatype): [polygons]}
    """
    ret = {}
    filterIsNotList = not all( [ type(filter) == list for filter in filters] )
    filterIsNotTuple = not all( [ type(filter) == tuple for filter in filters] )
    filterHasNot2Items = not all( [ len(filter) == 2 for filter in filters] )
    if filterIsNotList and filterIsNotTuple and filterHasNot2Items:
        raise TypeError("Each filter must be a list of tuples or lists of length 2!")
    for filter in filters:
        layer, datatype = filter
        if (layer, datatype) in ret.keys():
            raise ValueError("The specifications must be unique!")
        ret[(layer, datatype)] = get_polygons_by_spec(cell, [filter])
    return ret
      
def check_polygon_in_cell(
    polygon,
    cell: Cell,
) -> bool:
    """_summary_
    Checks if a polygon is inside a cell already
    Args:
        polygon (_type_): _description_
        cell (Cell): _description_

    Returns:
        bool: _description_
    """
    # check if the received polygons are valid
    if type(polygon) != Polygon and type(polygon) != RobustPath:
        raise TypeError("Polygon must be a Polygon or RobustPath object!")
    filters = [(polygon.layer, polygon.datatype)]
    for other in get_polygons_by_spec(cell, filters):
        if check_same_polygon(polygon, other):
            return True
    return False

def check_via_connection(
    polyA,
    via,
    polyB
) -> bool:
    """_summary_
    Checks if the polygon A and B are connected through
    the specified via
    Args:
        polyA   (Polygon): polygon in metal layer above
        via     (Polygon): polygon in metal layer between
        polyB   (Polygon): polygon in metal layer below
    Returns:
        bool: True if the polygons are connected through the via
                False otherwise
    Notes: 
        # ! This function assumes that the polygons A, via and B
        # ! are in three consecutive gds layers! If they are not, the provided
        # ! result will not mean anything
    """
    # check if the received polygons are valid
    if type(polyA) != Polygon and type(polyA) != RobustPath:
        raise TypeError("PolyA must be a Polygon or RobustPath object!")
    if type(polyB) != Polygon and type(polyB) != RobustPath:
        raise TypeError("PolyB must be a Polygon or RobustPath object!")
    if type(via) != Polygon and type(via) != RobustPath:
        raise TypeError("Via must be a Polygon or RobustPath object!")
    # check if the polygons are the same in pairs
    if check_same_polygon(polyA, polyB):
        raise ValueError("PolyA and PolyB must be different!")
    if check_same_polygon(polyA, via):
        raise ValueError("PolyA and Via must be different!")
    if check_same_polygon(polyB, via):
        raise ValueError("PolyB and Via must be different!")
    if polyA.layer == polyB.layer and polyA.datatype == polyB.datatype:
        raise ValueError("PolyA and PolyB must be in different layers!")
    if polyA.layer == via.layer and polyA.datatype == via.datatype:
        raise ValueError("PolyA and Via must be in different layers!")
    if polyB.layer == via.layer and polyB.datatype == via.datatype:
        raise ValueError("PolyB and Via must be in different layers!")
    # finally , check for mutual overlap of vias by the two polygons
    return bool_polygon_overlap_check(polyA,via) and bool_polygon_overlap_check(polyB,via)


def join_overlapping_polygons_cell(
    cell: Cell,
    layerMap: dict,
) -> Cell:
    """_summary_
    Joins overlapping the overlapping polygons for each metal 
    layer of a gds cell (resembling a layout), in order to
    join multiple intercepting polygons in a single polygon.
    Args:
        cell        (Cell)  : the gdspy.Cell object
        layerMap    (dict)  : dictionary of {"layer name": (layer, datatype)} tuples
    Returns:
        Cell: the new gdspy.Cell object with the joined polygons
    """
    newCell = Cell(cell.name+"_joined")
    polygons = get_polygon_dict(cell, list(layerMap.values()))
    for layer, datatype in polygons.keys():
        poly = boolean( polygons[(layer, datatype)], polygons[(layer, datatype)], 'or', layer = layer, datatype = datatype )
        [newCell.add(p) for p in poly]
    return newCell


def fuse_overlapping_cells(
    cellA: Cell,
    cellB: Cell
) -> Cell:
    """_summary_
    Fuses both cells into a single cell if both cells
    have at least one polygon of any layer in common
    Args:
        cellA       (Cell): gdspy Cell object
        cellB       (Cell): gdspy Cell object
        maxPoints   (int, optional): Maximum number of points inside polys when uniting cells. Defaults to 199.
        precision (float, optional): Precision of the cuts when performing union of cells.Defaults to 1e-3.

    Returns:
        Cell: Cells resulting from the union of cellA and cellB
        None: If no polygon of any layer in common
    """
    # get the layers that are common to both cells
    layersA = [poly.layer for poly in cellA.get_polygons()]
    datatypesA = [poly.datatype for poly in cellA.get_polygons()]
    ldA = list(zip(layersA, datatypesA))
    layersB = [poly.layer for poly in cellB.get_polygons()]
    datatypesB = [poly.datatype for poly in cellB.get_polygons()]
    ldB = list(zip(layersB, datatypesB))
    commonLd = [ld for ld in ldA if ld in ldB] # get the common layer,datatype tuples to both cells
    for polyA in get_polygons_by_spec(cellA, commonLd):
        if check_polygon_in_cell(polyA, cellB):
            # fuse the cells together and return it
            fuseCell = cellA.copy(cellA.name)
            [fuseCell.add(item) for item in itertools.chain(cellB.get_polygons(), cellB.get_paths(), cellB.get_labels())]
            [fuseCell.add(Reference(ref)) for ref in cellB.dependencies(False)]
            return fuseCell
    return None

def select_abstraction_depth(
    name: str,
    cell: Cell,
    depth: int = None,
    filters: list = []
) -> Cell:
    """_summary_
    Extracts the selected layers of the 
    cells of the library at a given 
    abstraction depth, and returns a new Library
    Args:
        name   (str)        : name of the returning library
        cell   (cell)       : Cell object to extract from
        depth  (int)        : abstraction depth
        select (dict)       : list of selected (layer, datatype) tuples
    Returns:
        Cell: the newly extracted cell from the depth parameters
        None:  if no polygons/paths are parsed from a cell to another
    NOTE : #! datatype keyword argument is not being supported in current gdstk.Cell.get_polygons function !! Report Problem
    """
    
    # check if the filters are valid
    filterIsNotList = not all( [ type(filter) == list for filter in filters] )
    filterIsNotTuple = not all( [ type(filter) == tuple for filter in filters] )
    filterHasNot2Items = not all( [ len(filter) == 2 for filter in filters] )
    if filterIsNotList and filterIsNotTuple and filterHasNot2Items:
        raise TypeError("Each filter must be a list of tuples or lists of length 2!")
    # create a new library
    newCell = Cell(name)
    newCell.properties = copy(cell.properties)
    if len(filters) > 0:
        for poly in itertools.chain(
            cell.get_polygons(depth = depth), 
            cell.get_paths( depth = depth ), 
            cell.get_labels(depth = depth )
        ):
            if (poly.layer, poly.datatype) in filters:
                newCell.add(poly)
    else: # copy all the items withou layer, datatype restriction, only applying depth restriction
        for poly in itertools.chain(
            cell.get_polygons(depth = depth), 
            cell.get_paths(depth = depth), 
            cell.get_labels(depth = depth)
        ):
            newCell.add(poly)
    return newCell if len(newCell.polygons) > 0 else None

def add_port(
    layout: Cell,
    table: GdsTable,
    name: str = "new_port",
    typo: str = "input",
    location: tuple = (0,0),
    width: float = 1.0,
    layer: str = "met1",
    resistance = 0.0,
) -> SpeedsterPort:
    """_summary_
    Adds/Draws a rectangular (square) port to the a layout cell
    Args:
        port        (SpeedsterPort) : SpeedsterPort object
        layout      (Cell)          : a gdspy Cell object containing the layout
                                    to which the port will be added
        table       (GdsTable)      : GdsTable object containing the GDS Table information
        name        (str)           : name of the port
        type        (str)           : type of the port (input, output or inout)
        location    (tuple)         : location of the port
        width       (float)         : width of the port
        layer       (str)           : name of the layer of the port
        resistance  (float)         : resistance of the port
    Returns:
        SpeedsterPort: the newly created SpeedsterPort object
    """
    port = SpeedsterPort(
        name = name,
        ioType = SpeedsterPortType(typo),
        resistance = resistance,
        location = location,
        width = width,
        layer = layer
    )
    l, dt = table.getGdsLayerDatatypeFromLayerNamePurpose(
        port.layer, 
        GdsLayerPurpose(port.purpose)
    )[0]
    portPoly = Polygon(port.get_polygon(), layer = l, datatype = dt)
    portPoly.set_property("SpdstrPort", True)
    portPoly.set_property("SpdstrPortName", port.name)
    portPoly.set_property("SpdstrPortType", port.ioType.value)
    portPoly.set_property("SpdstrPortResistance", port.resistance)
    layout.add(portPoly)
    return port