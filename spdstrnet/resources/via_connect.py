import gdspy
lib = gdspy.GdsLibrary()
lib.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/via_connect.gds")

colours = {
    "white": "#FFFFFF",
    "black": "#020202"
} 
gdspy.LayoutViewer(lib, background = colours["white"])