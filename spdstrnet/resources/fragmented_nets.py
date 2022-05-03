import gdspy

lib = gdspy.GdsLibrary()
# import fragmented crossed metal net
lib.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/fragmented_net.gds")

colours = {
    "white": "#FFFFFF",
    "black": "#020202"
} 
gdspy.LayoutViewer(lib, background = colours["white"])