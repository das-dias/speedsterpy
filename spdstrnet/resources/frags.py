import gdspy
lib = gdspy.GdsLibrary()
lib.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/frags.gds")

colours = {
    "white": "#FFFFFF",
    "black": "#020202"
} 
gdspy.LayoutViewer(lib, background = colours["white"])