import unittest
import sys
import gdstk
import numpy as np
from loguru import logger
from spdstrnet import (
    check_point_inside_polygon,
    check_polygon_overlap,
    check_polygon_overlap,
    bool_polygon_overlap_check,
    get_common_edges,
    check_neighbour_polygons,
    check_same_polygon,
    check_polygon_contains_polygon,
    find_centroid,
    unit_vector,
    saturate_vector,
    check_neighbour_direction,
    PolygonDirection,
    get_rectangle_length_width_ratio,
    fragment_polygon,
    fragment_net,
    get_polygons_by_spec,
    get_polygon_dict,
    check_polygon_in_cell,
    check_via_connection,
    join_overlapping_polygons_cell,
    fuse_overlapping_cells,
    select_abstraction_depth,
    add_port
)
from spdstrnet import SpeedsterPort
from spdstrutil import (
    GdsTable,
    GdsLayerPurpose,
    readGdsTable
)
class TestGeometry(unittest.TestCase):
    ld = {
        "met1": {"layer":1, "datatype": 0},
        "met2": {"layer":2, "datatype": 0},
        "via": {"layer":3, "datatype": 0},
    }
    def test_check_point_inside_polygon(self):
        point = [.3, .3]
        point2 = [3.1, 3.1]
        poly = gdstk.rectangle( (0.0,0.0),(3.0,1.0), **self.ld["met1"]) 
        self.assertTrue( check_point_inside_polygon(poly, point) )
        self.assertFalse( check_point_inside_polygon(poly, point2) )
    
    def test_check_polygon_overlap(self):
        poly    = gdstk.rectangle( (0.0,0.0),(3.0,1.0), **self.ld["met1"])
        poly2   = gdstk.rectangle( (0.0,0.0),(1.0,3.0), **self.ld["met1"])
        poly3   = gdstk.rectangle( (0.0,0.0),(1.0,1.0), **self.ld["met2"])
        poly4   = gdstk.rectangle( (1.0,0.0),(2.0,1.0), **self.ld["met2"])
        
        self.assertIsNotNone( check_polygon_overlap(poly, poly2) )
        self.assertIsNotNone( check_polygon_overlap(poly, poly3) )
        self.assertIsNone( check_polygon_overlap(poly3, poly4) )
        self.assertIsNone( check_polygon_overlap(poly2, poly4) )
        

    def test_bool_polygon_overlap_check(self):
        poly    = gdstk.rectangle( (0.0,0.0),(3.0,1.0), **self.ld["met1"])
        poly2   = gdstk.rectangle( (0.0,0.0),(1.0,3.0), **self.ld["met1"])
        poly3   = gdstk.rectangle( (0.0,0.0),(1.0,1.0), **self.ld["met2"])
        poly4   = gdstk.rectangle( (1.0,0.0),(2.0,1.0), **self.ld["met2"])
        
        self.assertTrue( bool_polygon_overlap_check(poly, poly2) )
        self.assertTrue( bool_polygon_overlap_check(poly, poly3) )
        self.assertFalse( bool_polygon_overlap_check(poly3, poly4) )
        self.assertFalse( bool_polygon_overlap_check(poly2, poly4) )
        
    
    def test_get_common_edges(self):
        poly    = gdstk.rectangle( (0.0,0.0),(3.0,1.0), **self.ld["met1"])
        poly2   = gdstk.rectangle( (0.0,0.0),(1.0,3.0), **self.ld["met1"])
        poly3   = gdstk.rectangle( (0.0,0.0),(1.0,1.0), **self.ld["met2"])
        poly4   = gdstk.rectangle( (1.0,0.0),(2.0,1.0), **self.ld["met2"])
        poly5   = gdstk.rectangle( (2.0,0.0),(4.0,4.0), **self.ld["met1"])
        poly6   = gdstk.rectangle( (0.0, 1.0), (1.0, 2.0), **self.ld["met2"] )
        poly7   = gdstk.rectangle( (1.0, 0.0), (2.0, 4.0), **self.ld["met2"] )
        poly8   = gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] )
        poly9   = gdstk.rectangle( (1.0, 1.0), (3.0, 3.0), **self.ld["met2"] )
        edges = {
            "t1": [
                [ (0.0, 0.0), (1.0, 0.0) ],
                [ (0.0, 1.0), (0.0, 0.0) ]
            ],
            "t2": [
                [ (1.0, 1.0), (1.0, 0.0) ],
            ]
        }
        self.assertIsNotNone( get_common_edges(poly, poly2) )
        self.assertEqual(
            get_common_edges(poly, poly2)[0],
            edges["t1"][0]
        )
        self.assertIsNotNone( get_common_edges(poly3, poly4) )
        self.assertEqual(
            get_common_edges(poly3, poly4)[0],
            edges["t2"][0]
        )
        self.assertIsNone( get_common_edges(poly3, poly5) )
        self.assertIsNotNone( get_common_edges(poly, poly5) )
        self.assertIsNotNone( get_common_edges(poly6, poly7) )
        self.assertIsNotNone( get_common_edges(poly8, poly9) )
        
    
    def test_check_neighbour_polygon(self):
        poly    = gdstk.rectangle( (0.0,0.0),(3.0,1.0), **self.ld["met1"])
        poly2   = gdstk.rectangle( (0.0,0.0),(1.0,3.0), **self.ld["met1"])
        poly3   = gdstk.rectangle( (0.0,0.0),(1.0,1.0), **self.ld["met2"])
        poly4   = gdstk.rectangle( (1.0,0.0),(2.0,1.0), **self.ld["met2"])
        poly5   = gdstk.rectangle( (2.0,0.0),(4.0,4.0), **self.ld["met1"])
        self.assertFalse( check_neighbour_polygons(poly, poly2) )
        self.assertTrue( check_neighbour_polygons(poly3, poly4) )
        self.assertFalse( check_neighbour_polygons(poly4, poly5) )
        self.assertFalse( check_neighbour_polygons(poly2, poly4) )
        
        
    
    def test_check_same_polygon(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tests = {
            "t1": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] ),
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] )
            ],
            "t2": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"]),
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] )
            ],
            "t3": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] ),
                gdstk.rectangle( (0.0, 0.0), (3.0, 1.0), **self.ld["met2"] )
            ],
            "t4": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"]),
                gdstk.Polygon( points, **self.ld["met2"] )
            ],
            "t5": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"]),
                gdstk.Polygon( points, **self.ld["met1"] )
            ]
        }
        self.assertTrue( check_same_polygon( tests["t1"][0], tests["t1"][1] ) )
        self.assertFalse( check_same_polygon( tests["t2"][0], tests["t2"][1] ) )
        self.assertFalse( check_same_polygon( tests["t3"][0], tests["t3"][1] ) )
        self.assertTrue( check_same_polygon( tests["t4"][0], tests["t4"][1] ) )
        self.assertFalse( check_same_polygon( tests["t5"][0], tests["t5"][1] ) )
    
    def test_check_polygon_contains_polygon(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tests = {
            "t1": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] ),
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] )
            ],
            "t2": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"]),
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] )
            ],
            "t3": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] ),
                gdstk.rectangle( (0.0, 0.0), (3.0, 1.0), **self.ld["met2"] )
            ],
            "t4": [
                gdstk.rectangle( (0.0, 0.0), (4.0, 4.0), **self.ld["met2"]),
                gdstk.rectangle( (1.0, 1.0), (3.0, 3.0), **self.ld["met2"])
            ],
            "t5": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"]),
                gdstk.rectangle( (1.0, 1.0), (3.0, 3.0), **self.ld["met2"])
            ]
        }
        self.assertTrue( check_polygon_contains_polygon( tests["t1"][0], tests["t1"][1] ) )
        self.assertTrue( check_polygon_contains_polygon( tests["t1"][1], tests["t1"][0] ) )
        self.assertFalse( check_polygon_contains_polygon( tests["t2"][0], tests["t2"][1] )  )
        self.assertFalse( check_polygon_contains_polygon( tests["t3"][0], tests["t3"][1] ) )
        self.assertTrue( check_polygon_contains_polygon( tests["t3"][1], tests["t3"][0] ) )
        self.assertTrue( check_polygon_contains_polygon( tests["t4"][0], tests["t4"][1] ) )
        self.assertFalse( check_polygon_contains_polygon( tests["t5"][0], tests["t5"][1] ) )
        self.assertFalse( check_polygon_contains_polygon( tests["t5"][1], tests["t5"][0] ) )
    
    def test_find_centroid(self):
        points = [
            [(0, 0), (1,0), (1,1)],
            [(0, 0), (1,0), (2,1), (1,3), (0,3), (-1,2)]
        ]
        polys = [
            gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] ),
            gdstk.Polygon( points[0] ),
            gdstk.Polygon( points[1] ),
        ]
        centroids = [
            (0.5, 0.5),
            ( np.mean([x for x,y in points[0]]), np.mean([y for x,y in points[0]]) ),
            ( np.mean([x for x,y in points[1]]), np.mean([y for x,y in points[1]]) )
        ]
        self.assertEqual( find_centroid( polys[0] ), centroids[0] )
        self.assertEqual( find_centroid( polys[1] ), centroids[1] )
        self.assertEqual( find_centroid( polys[2] ), centroids[2] )
        
    
    def test_unit_vector(self):
        edges = [
            [(0,0), (1,0)],
            [(0,0), (0,1)],
            [(0,0), (1,1)],
            [(0,0), (3,0)],
            [(0,0), (0,3)],
            [(-4,0), (0,0)],
            [(0,0), (0,-4)],
        ]
        expected_vecs = [
            [1,0],
            [0,1],
            [np.sqrt(2)/2, np.sqrt(2)/2],
            [1,0],
            [0,1],
            [1,0],
            [0,-1],
        ]
        self.assertEqual( unit_vector( edges[0][0], edges[0][1] ), expected_vecs[0] )
        self.assertEqual( unit_vector( edges[1][0], edges[1][1] ), expected_vecs[1] )
        #! self.assertEqual( unit_vector( edges[2][0], edges[2][1] ), expected_vecs[2] ) # - fails due to floating point imprecision
        self.assertAlmostEqual( unit_vector( edges[2][0], edges[2][1] )[0], expected_vecs[2][0] )
        self.assertAlmostEqual( unit_vector( edges[2][0], edges[2][1] )[1], expected_vecs[2][1] )
        self.assertEqual( unit_vector( edges[3][0], edges[3][1] ), expected_vecs[3] )
        self.assertEqual( unit_vector( edges[4][0], edges[4][1] ), expected_vecs[4] )
        self.assertEqual( unit_vector( edges[5][0], edges[5][1] ), expected_vecs[5] )
        self.assertEqual( unit_vector( edges[6][0], edges[6][1] ), expected_vecs[6] )
        
    def test_saturate_vector(self):
        vecs = [
            [0.8,0.3],
            [0,1],
            [-0.6, 0.5],
            [0.5,0],
            [0,-0.4],
            [-0.8,0],
            [0,-0.5],
        ]
        expected_vecs = [
            [1,0],
            [0,1],
            [-1, 0],
            [0,0],
            [0,0],
            [-1,0],
            [0,0],
        ]
        self.assertEqual( saturate_vector( vecs[0] ), expected_vecs[0] )
        self.assertEqual( saturate_vector( vecs[1] ), expected_vecs[1] )
        self.assertEqual( saturate_vector( vecs[2] ), expected_vecs[2] )
        self.assertEqual( saturate_vector( vecs[3] ), expected_vecs[3] )
        self.assertEqual( saturate_vector( vecs[4] ), expected_vecs[4] )
        self.assertEqual( saturate_vector( vecs[5] ), expected_vecs[5] )
        self.assertEqual( saturate_vector( vecs[6] ), expected_vecs[6] )
    
    def test_check_neighbour_direction(self):
        polys = {
            "t1": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met1"] ),
                gdstk.rectangle( (1.0, 0.0), (2.0, 1.0), **self.ld["met1"] )
            ],
            "t2": [
                gdstk.rectangle( (1.0, 0.0), (2.0, 1.0), **self.ld["met1"] ),
                gdstk.rectangle( (1.0, 1.0), (2.0, 2.0), **self.ld["met1"] )
            ],
            "t3": [
                gdstk.rectangle( (0.0, 1.0), (1.0, 2.0), **self.ld["met2"] ),
                gdstk.rectangle( (1.0, 0.0), (2.0, 4.0), **self.ld["met2"] )
            ],
            "t4": [
                gdstk.rectangle( (0.0, 0.0), (4.0, 4.0), **self.ld["met2"] ),
                gdstk.rectangle( (1.0, 1.0), (3.0, 3.0), **self.ld["met2"] )
            ],
            "t5": [
                gdstk.rectangle( (0.0, 0.0), (1.0, 1.0), **self.ld["met2"] ),
                gdstk.rectangle( (1.0, 1.0), (3.0, 3.0), **self.ld["met2"] )
            ],
            "t6": [
                gdstk.Polygon( [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], **self.ld["met2"] ),
                gdstk.Polygon( [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], **self.ld["met2"] )
            ]
        }
        expected = [
            PolygonDirection.EAST,
            PolygonDirection.NORTH,
            PolygonDirection.EAST,
            None,
            PolygonDirection.NORTH_EAST,
            PolygonDirection.NORTH_EAST
        ]
        self.assertEqual( check_neighbour_direction( polys["t1"][0], polys["t1"][1] ), expected[0] )
        self.assertEqual( check_neighbour_direction( polys["t2"][0], polys["t2"][1] ), expected[1] )
        self.assertEqual( check_neighbour_direction( polys["t3"][0], polys["t3"][1] ), expected[2] )
        self.assertEqual( check_neighbour_direction( polys["t4"][0], polys["t4"][1] ), expected[3] )
        self.assertEqual( check_neighbour_direction( polys["t5"][0], polys["t5"][1] ), expected[4] )
        self.assertEqual( check_neighbour_direction( polys["t6"][0], polys["t6"][1] ), expected[5] )
        
    
    def test_get_rectangle_length_width_ratio(self):
        rects = [
            gdstk.rectangle( (0.0, 0.0), (2.0, 1.0), **self.ld["met1"] ),
            gdstk.rectangle( (0.0, 0.0), (2.0, 1.0), **self.ld["met1"] ),
            gdstk.rectangle( (0.0, 0.0), (4.0, 1.0), **self.ld["met1"]).rotate(np.pi/4),
            gdstk.rectangle( (0.0, 0.0), (4.0, 1.0), **self.ld["met1"]).rotate(np.pi/4)
        ]
        dirs = [
            PolygonDirection.EAST,
            PolygonDirection.SOUTH,
            PolygonDirection.NORTH_EAST,
            PolygonDirection.SOUTH_EAST
        ]
        expected = [
            2.0,
            1/2,
            4.0,
            1/4
        ]
        self.assertAlmostEqual( get_rectangle_length_width_ratio(rects[0], dirs[0]), expected[0] )
        self.assertAlmostEqual( get_rectangle_length_width_ratio(rects[1], dirs[1]), expected[1] )
        self.assertAlmostEqual( get_rectangle_length_width_ratio(rects[2], dirs[2]), expected[2] )
        self.assertAlmostEqual( get_rectangle_length_width_ratio(rects[3], dirs[3]), expected[3] )
        
    
    def test_fragment_polygon(self):
        lib = gdstk.Library()
        cell = lib.new_cell("frags")
        rect = gdstk.rectangle((0,0), (5, 1), **self.ld["met1"])
        frags = fragment_polygon(rect)
        self.assertIsNotNone(rect)
        poly = gdstk.Polygon([
                (0,0),
                (5,0),
                (5,5),
                (4,5),
                (4,1),
                (0,1)
            ],
            **self.ld["met2"]
        )
        frags_poly = fragment_polygon(poly)
        self.assertIsNotNone(frags_poly)
        layer = self.ld["met1"]["layer"]
        for poly in frags_poly:
            poly.layer = layer
            cell.add(poly)
            layer += 1
        for poly in frags:
            poly.layer = layer
            cell.add(poly)
            layer += 1
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/frags.gds")
        
    def test_fragment_net(self):
        # get gds table
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTable = readGdsTable(path)
        # get interconnect net
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal.gds")
        net = [cell for cell in lib.cells if cell.name == "crossed_metal"][0]
        fragmented_net = fragment_net("fragmented_crossed_metal", net, gdsTable)
        self.assertIsNotNone(fragmented_net)
        self.assertEqual(type(fragmented_net), gdstk.Cell)
        lib.add(fragmented_net)
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/fragmented_net.gds")
        
    
    def test_get_polygons_by_spec(self):
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal.gds")
        net = [cell for cell in lib.cells if cell.name == "crossed_metal"][0]
        # filter
        filter = [(68,20)] 
        polys = get_polygons_by_spec(net, filter)
        self.assertTrue( all( [(p.layer,p.datatype) == (68,20) for p in polys] ) )
    
    def test_get_polygons_dict(self):
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal.gds")
        net = [cell for cell in lib.cells if cell.name == "crossed_metal"][0]
        # filter
        filter = [(68,20), (69,44)] 
        polys = get_polygon_dict(net, filter)
        self.assertTrue( (68,20) in polys.keys() )
        self.assertTrue( (69,44) in polys.keys() )
        self.assertTrue( all( [(p.layer,p.datatype) == (68,20) for p in polys[(68,20)]] ) )
        
    def test_check_polygon_in_cell(self):
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal.gds")
        net = [cell for cell in lib.cells if cell.name == "crossed_metal"][0]
        # filter
        filter = [(68,20)] 
        poly = get_polygons_by_spec(net, filter)[0]
        self.assertTrue( check_polygon_in_cell(poly, net) )
    
    def test_check_via_connection(self):
        polyA = gdstk.rectangle((0,0), (1,6), **self.ld["met1"])
        polyB = gdstk.rectangle((0,5), (6,6), **self.ld["met2"])
        via = gdstk.rectangle((5.2,5.2), (5.8,5.8), **self.ld["via"])
        polyC = gdstk.rectangle((5,0), (6,6), **self.ld["met1"])
        lib = gdstk.Library()
        cell = lib.new_cell("via_connect")
        cell.add(polyA)
        cell.add(polyB)
        cell.add(via)
        cell.add(polyC)
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/via_connect.gds")
        self.assertFalse( check_via_connection(polyA, via, polyB) )
        self.assertTrue( check_via_connection(polyC, via, polyB) )
        
    def test_join_overlapping_pollygons_cell(self):
        # get gds table
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTable = readGdsTable(path)
        # get interconnect net
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/slotted_metal.gds")
        net = [cell for cell in lib.cells if cell.name == "slotted_metal"][0]
        united_cell = join_overlapping_polygons_cell(net, gdsTable.getDrawingMetalLayersMap())
        self.assertIsNotNone(united_cell)
        self.assertEqual(type(united_cell), gdstk.Cell)
        lib.add(united_cell)
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/slotted_metal_united.gds")
    
    def test_fuse_overlapping_cells(self):
        polys1 = [
            gdstk.rectangle((0,5), (6,6), **self.ld["met2"]),
            gdstk.rectangle((5.2,5.2), (5.8,5.8), **self.ld["via"]),
            gdstk.rectangle((5,0), (6,6), **self.ld["met1"])
        ]
        polys2 = [
            gdstk.rectangle((0,5), (6,6), **self.ld["met2"])
        ]
        polys3 = [
            gdstk.rectangle((0,0), (1,6), **self.ld["met2"]),
        ]
        cell1 = gdstk.Cell("cell1")
        cell2 = gdstk.Cell("cell2")
        cell3 = gdstk.Cell("cell3")
        for poly in polys1:
            cell1.add(poly)
        for poly in polys2:
            cell2.add(poly)
        for poly in polys3:
            cell3.add(poly)
        fused_cell = fuse_overlapping_cells(cell1, cell2)
        fused_cell.name = "fused_cell"
        self.assertIsNotNone(fused_cell)
        fused_cell2 = fuse_overlapping_cells(cell1, cell3)
        self.assertIsNone(fused_cell2)
        lib = gdstk.Library()
        lib.add(fused_cell)
        lib.add(cell1)
        lib.add(cell2)
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/fused_cells.gds")    
    
    def test_select_abstraction_depth(self):
        # get slotted metal 1 interconnect net
        lib2 = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/slotted_metal.gds")
        slotted_net = [cell for cell in lib2.cells if cell.name == "slotted_metal"][0]
        # get crossed metal 1 / 2 interceonnect net
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal.gds")
        crossed_net = [cell for cell in lib.cells if cell.name == "crossed_metal"][0]
        # add slotted metal 1 interconnect net to crossed metal net as a reference, with its origin placed in (10,0) and magnified by half
        slotted_ref = gdstk.Reference(slotted_net, (9.6,0), magnification = 1.5)
        crossed_net.add(slotted_ref)
        # get abstraction depth
        selected_cells = [
            select_abstraction_depth( "depth1_crossed", crossed_net, depth = 0, filters = [(69,20)] ),
            select_abstraction_depth( "depth2_crossed", crossed_net, depth = 1, filters = [(69,20)] ),
            select_abstraction_depth( "depth3_crossed", crossed_net ),
            select_abstraction_depth( "depth4_crossed", crossed_net, depth = 0, filters = [(69,20)] ),
            select_abstraction_depth( "depth5_crossed", crossed_net, filters = [(70,20)] )
        ]
        self.assertIsNotNone(selected_cells[0])
        self.assertEqual(type(selected_cells[0]), gdstk.Cell)
        self.assertIsNotNone(selected_cells[1])
        self.assertEqual(type(selected_cells[1]), gdstk.Cell)
        self.assertIsNotNone(selected_cells[2])
        self.assertEqual(type(selected_cells[2]), gdstk.Cell)
        self.assertIsNotNone(selected_cells[3])
        self.assertEqual(type(selected_cells[3]), gdstk.Cell)
        self.assertIsNone(selected_cells[4])
        for cell in selected_cells[:-1]:
            lib.add(cell)
        # save new lib
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal_slotted_ref.gds")
        
    def test_add_port(self):
        # get gds table
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTable = readGdsTable(path)
        # get net
        lib = gdstk.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal_slotted_ref.gds")
        net = [cell for cell in lib.cells if cell.name == "depth3_crossed"][0]
        # create a new speedster port 
        port = add_port(
            net,
            gdsTable,
            name = "port1",
            typo = "input",
            location = (14.5,1.1),
            width = 0.25,
            layer = "met1",
            resistance = 1e-3
        )
        l, dt = gdsTable.getGdsLayerDatatypeFromLayerNamePurpose(
                    port.layer, 
                    GdsLayerPurpose(port.purpose)
                )[0]
        self.assertIsNotNone(port)
        self.assertTrue(isinstance(port, SpeedsterPort))
        self.assertIsNotNone( get_polygons_by_spec(net, filters = [(l,dt)]) )
        # write new lib in memory to observe it
        lib.write_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal_slotted_ref_withport.gds")
        # observed that the port is being added successfully
        
if __name__ == '__main__':
    unittest.main()