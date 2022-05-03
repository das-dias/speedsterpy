import gdspy
ld = {
    "met1": {"layer":1, "datatype": 0},
    "met2": {"layer":2, "datatype": 0}
}
lib = gdspy.GdsLibrary()
lib.read_gds("/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrnet/resources/data/crossed_metal_slotted_ref_withport.gds")
colours = {
    "white": "#FFFFFF",
    "black": "#020202"
} 
gdspy.LayoutViewer(lib, background = colours["white"])
