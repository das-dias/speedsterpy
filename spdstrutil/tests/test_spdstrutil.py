import sys
import unittest
sys.path.append("../spdstrutil")
from spdstrutil import (
    __version__,
    GdsLayerPurpose,
    GdsTable,
    readGdsTable,
    writeGdsTable,
    timer,
)

class TestUtil(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, '0.1.2')

    def test_gds_table_read(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        self.assertEqual(type(gdsTab), GdsTable)
        self.assertIsNotNone(gdsTab)
        self.assertEqual(
            gdsTab.getLayerPurpose(71,20), # met4, drawing, text
            [GdsLayerPurpose.DRAWING, GdsLayerPurpose.TEXT]
        )
        
    def test_gds_table_write(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrutil/resources"
        gdsTab = GdsTable()
        gdsTab.add(
            layer = 71,
            dataType = 20,
            name = "met4",
            purpose = [GdsLayerPurpose.DRAWING.name, GdsLayerPurpose.TEXT.name],
            description = "Metal 4"
        )
        self.assertIsNone(writeGdsTable(gdsTab, path))

    def test_gds_table_read_yaml(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrutil/resources/gds_table.yaml"
        gdsTab = readGdsTable(path)
        #print(str(gdsTab))
        self.assertEqual(
            gdsTab.getLayerPurpose(71,20), # met4, drawing, text
            [GdsLayerPurpose.DRAWING, GdsLayerPurpose.TEXT]
        )
        self.assertEqual(
            gdsTab.table[(71,20)], # met4, drawing, text
            {
                "name": "met4",
                "purpose": [GdsLayerPurpose.DRAWING, GdsLayerPurpose.TEXT],
                "description": "Metal 4"
            }
        )

    def test_filter_gds_table_by_purpose(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        #print(str(gdsTab))
        newTab = gdsTab.getGdsTableEntriesFromPurpose(GdsLayerPurpose.DRAWING)
        #print(str(newtab))
        self.assertEqual(
            newTab.getLayerName(71,20), # met4, drawing, text
            "met4"
        )
        self.assertEqual(
            newTab.getLayerPurpose(71,20), # met4, drawing, text
            [GdsLayerPurpose.DRAWING, GdsLayerPurpose.TEXT]
        )
        self.assertIsNone(
            newTab.getGdsTableEntriesFromPurpose(GdsLayerPurpose.BLOCK)
        )
        

    def test_filter_gds_table_by_layer_name(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        #print(str(gdsTab))
        newtab = gdsTab.getGdsTableEntriesFromLayerName("met4")
        #print(str(newtab))
        self.assertEqual(
            newtab.getLayerName(71,20), # met4, drawing, text
            "met4"
        )
        self.assertIsNone(
            newtab.getGdsTableEntriesFromLayerName("met5")
        )

    def test_get_gds_layer_datatype_from_layer_name_purpose(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        #print(str(gdsTab))
        layerDatatype = gdsTab.getGdsLayerDatatypeFromLayerNamePurpose("met4", GdsLayerPurpose.NET)
        self.assertIn(
            (71,23),
            layerDatatype
        )

    def test_get_metal_drawing_layers(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        layerMap = gdsTab.getDrawingMetalLayersMap()
        self.assertEqual(
            layerMap["met4"],
            (71,20)
        )
        self.assertEqual(
            layerMap["via3"],
            (70,44)
        )
        self.assertIsNone(
            layerMap.get("pad")
        )
        self.assertIsNone(
            layerMap.get("mcon")
        )

    def test_add_backannotation(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        gdsTab = gdsTab.addBackannotation()
        self.assertEqual(
            gdsTab.getGdsLayerDatatypeFromLayerNamePurpose(
                "met4",
                GdsLayerPurpose.BACKANNOTATION    
            )[0],
            (71,26)
        )
        self.assertIsNone(
            gdsTab.getGdsLayerDatatypeFromLayerNamePurpose(
                "mcon",
                GdsLayerPurpose.BACKANNOTATION    
            )
        )
        

    def test_add_highlightnet(self):
        path = "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/resources/sky130/skywater-pdk-libs-sky130_fd_pr/skywater130/gds_layers.csv"
        gdsTab = readGdsTable(path)
        gdsTab = gdsTab.addHighlight()
        self.assertEqual(
            gdsTab.getGdsLayerDatatypeFromLayerNamePurpose(
                "highlight",
                GdsLayerPurpose.HIGHLIGHTING    
            )[0],
            (65,0)
        )
        self.assertIsNone(
            gdsTab.getGdsLayerDatatypeFromLayerNamePurpose(
                "mcon",
                GdsLayerPurpose.HIGHLIGHTING
            )
        )
        
    def test_timer(self):
        @timer
        def wait_ms(msec: float) -> str:
            import time
            time.sleep(msec*1e-3)
            return "Waited {} milli seconds".format(msec)
        self.assertEqual(wait_ms(10), "Waited 10 milli seconds")
        
if __name__ == '__main__':
    unittest.main()