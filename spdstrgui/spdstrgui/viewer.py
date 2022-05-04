"""_summary_
viewer.py contains the main functionalities for the
plotting and interaction of the extracted resistance network
heatmap with the user of the main application - speedster

[author]    Diogo André Silvares Dias
[date]      2022-04-17
[contact]   das.dias@campus.fct.unl.pt
"""

# SECOND: AFTER COMPLETING ALL THE DATA STRUCTURES
# OF THE data.py FILE
# TODO: Develop a SpeedsterLayoutViewer class to 
# enable the multi layer png plotting of 
# selectable metal layers of the layout
# as well as the resitance heatmap generated from
# the point to point resistance extraction

# TODO : URGENTLY - TO ENABLE VISUAL TESTING OF SPDSTRRES MODULE
# white background
# original colour schemes for the layers / datatypes
# support to backannotation of resistance and ports must be added
# as features to the new viewer...

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from enum import Enum
import os
import re
import colorsys
import warnings
import numpy as np
# tkinter provides the GUI making tools
import tkinter
import tkinter.messagebox
import tkinter.colorchooser
# matplotlib provides the framework
# for obtaining a colour map for the
# resistance, current density or 
# dissipated power quantities
import matplotlib as mpl
import matplotlib.pylab as plt 
from matplotlib import cm # provides colour to value mapping
from matplotlib.ticker import ScalarFormatter # provides normalization of values between a scale of 0 and 1

from gdspy import (
    ColorDict,
    PatternDict,
    GdsLibrary,
    GdsCell,
)

_tkinteranchors = [
    tkinter.NW,
    tkinter.N,
    tkinter.NE,
    None,
    tkinter.W,
    tkinter.CENTER,
    tkinter.E,
    None,
    tkinter.SW,
    tkinter.S,
    tkinter.SE,
]

class PhysicalQuantities(Enum):
    """_summary_
    An enumeration of the possible physical quantities
    that the speedster app can plot for now
    Args:
        RESISTANCE : The extracted resistance of the network
        CURRENT_DENSITY : The extracted current density along network
        DISSIPATED_POWER : The extracted dissipated power along the network
    """
    RESISTANCE      = "Resistance"
    CURRENT_DENSITY = "Current Density"
    DISSIPATED_POWER= "Dissipated Power"
    
class ColorMapper(object):
    """_summary_
    A wrapper for the matplotlib colour mapping 
    framwork to ease the retreival of a colour 
    through a given physical quantity.
    
    This colour mapper object allows to quickly obtain
    rgb, bgr and hexcode for the colours associated with each
    resistance, current density or power dissipation value 
    computed for a given node during extraction.
    
    Args:
        cmapName (str, optional): 
            The string identifier of the colour map. Defaults to "seismic".
            
        cmap     (matplotlib.colors.Colormap): 
            The colour map object belonging to matplotlib
            
        minVal   (float): 
            The minimum value of the physical quantity. Defaults to 0.0.
        
        maxVal   (float):
            The maximum value of the physical quantity. Defaults to 1.0.
        
        quantity (str):
            The name of the physical quantity. Defaults to "".
        
        scale   (matplotlib.colors.Normalize):
            The normalization scale object to normalize physical quantity values
        
        scalarMap (matplotlib.cm.ScalarMappable):
            The scalar mapper object to link a value to a colour value
    """
    def __init__(
        self, 
        cmapName: str = "seismic",
        minVal: float = 0.0,
        maxVal: float = 1.0,
        quantity: str = "",
    ):
        """_summary_
        Class constructor
        Args:
            cmapName    (str, optional)     : namme of the colour map. 
                                                Defaults to "seismic" - the official Speedster colours
            minVal      (float, optional)   : minimum physical quantity value. Defaults to 0.0.
            maxVal      (float, optional)   : maximum physical quantity value. Defaults to 1.0.
            quantity    (str, optional)     : name of the physical quantity. Defaults to "".
        """
        super(ColorMapper, self).__init__()
        self.cmapName = cmapName
        self.cmap = plt.get_cmap(cmapName)
        self.minVal = minVal
        self.maxVal = maxVal
        self.quantity = quantity
        #set up colour map
        self.scale = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
        self.scalarMap = cm.ScalarMappable(norm=self.scale, cmap=self.cmap)
    
    def overwrite(
        self,
        cmapName: str = "seismic",
        minVal: float = 0.0,
        maxVal: float = 1.0,
        quantity: str = "",
    ) -> ColorMapper:
        """_summary_
        Overwrites the current colour map with a new one
        Args:
            cmapName (str, optional): _description_. Defaults to "seismic".
            minVal (float, optional): _description_. Defaults to 0.0.
            maxVal (float, optional): _description_. Defaults to 1.0.
            quantity (str, optional): _description_. Defaults to "".
        Returns:
            ColorMapper: _description_
        """
        self.__init__(cmapName, minVal, maxVal, quantity)
        return self
    
    def getRgba(
        self,
        val: float
    ) -> tuple:
        """_summary_
        Returns the RGB,Alpha (opacity) colour values
        from a given physical quantity value
        Args:
            val (float): physical quantity value
        Returns:
            (Red, Green, Blue, Alpha): colour values between [0 , 255]
        """
        rgbaArray = self.scalarMap.to_rgba(val)
        r, g, b, a = rgbaArray
        return (round(r*255), round(g*255), round(b*255), round(a*255))
    
    def getBgr(
        self,
        val: float,
    ) -> tuple:
        """_summary_
        Returns the BGR colour values
        from a given physical quantity value
        Args:
            val (float): physical quantity value
        Returns:
            (Blue, Green, Red): colour values between [0 , 255]
        """
        r, g ,b, _ = self.getRgba(val)
        return (b, g, r)
    
    def getRgb(
        self,
        val: float,
    ) -> tuple:
        """_summary_
        Returns the RGB colour values
        from a given physical quantity value
        Args:
            val (float): physical quantity value
        Returns:
            (Red, Green, Blue): colour values between [0 , 255]
        """
        b, g, r = self.getBgr(val)
        return (r , g, b)
    
    def getHexCode(
        self,
        val: float,
        keepAlpha: bool = False,
    ) -> str:
        """_summary_
        Returns the RGB,Alpha (opacity) hex code of the colour value
        from a given physical quantity value
        Args:
            val         (float)         : Physical quantity value
            keepAlpha   (bool, optional): Keep the alpha value out or in of the hexcode.
                                            Defaults to False.
        Returns:
            (Red, Green, Blue, Alpha): colour values between [0 , 255]
        """
        rgbaArray = self.scalarMap.to_rgba(val)
        return mpl.colors.to_hex(rgbaArray, keep_alpha=keepAlpha)
    
class BackannotationDict(dict):
    """_summary_
    A dictionary that stores the color to 
    float value mapping in order to back annotate
    the resistance, current density and power dissipation
    back on the layout
    Args:
        dict (dict): 
            Colour mapping dictionary of tuples : { float_value(float) : "colour hex code"(str) }
    """
    def __init__(self, default = None, cmap: ColorMapper = None):
        """_summary_
        Initializes the BackannotationDict class
        Args:
            default (_type_): _description_
            minVal (float, optional): Min physical unit value. Defaults to 0.0.
            maxVal (float, optional): Max physical unit value. Defaults to 1.0.
        Notes: 
            -   minVal and maxVal are used to establish the minimum and maximum value
                of the keys of the dict, defining the colour range of the backannotation
                colouring system.
        """
        super(BackannotationDict, self).__init__()
        self.default = default
        self.cmap = cmap
    
    def __missing__(self, key) -> str:
        if self.cmap is None:
            raise ValueError("No colour map has been set for the BackannotationDict")
        # if no default value is set, generate it from the colour map
        # key holds the physical quantity value
        rgb = self.cmap.getHexCode(key) if self.default is None else self.default
        self[key] = rgb
        return rgb

    def __getitem__(self, key) -> str:
        if key in self:
            return super(BackannotationDict, self).__getitem__(key)
        return self.__missing__(key)

class SpeedsterLayoutViewer(tkinter.Frame):
    """_summary_
    Speedster application dedicated GUI
    for the visualization of the integrated circuit
    layout.
    
    This GUI is strongly inspired by heitzmann's
    gdspy package LayoutViewer object.
    
    gdspy package LayoutViewer description:

        You can zoom in or out using control plus the mouse wheel, or drag a
        rectangle on the window with the 1st mouse button to zoom into that
        area.
        A ruler is available by clicking the 1st mouse button anywhere on
        the view and moving the mouse around.  The distance is shown in the
        status area.
        Double-clicking on any polygon gives some information about it.
        Color and pattern for each layer/datatype specification can be
        changed by left and right clicking on the icon in the layer/datatype
        list.  Left and right clicking the text label changes the
        visibility.
    
    Additional fucntionalities contain a button for the addition of
    input and output ports in order to signal the starting and ending point for
    the performing of resistance extraction. Aditional widgets include an
    integrated console for running python commands.
    
    There is also an additional possibility of parsing a BackannotationDict
    for, in adition with a parsed ResistanceMap allow for coloured back annotation
    of each polygon according to its resistance value.
    
    Args:    
        super (tkinter.Frame): Parent GUI framework object from TKinter package
        
        library (GdsLibrary): 
            Library object containing the integrated circuit cells
        
        cells (dict):
            Dictionary of {"name": GdsCell} tuples containing the integrated circuit cells
        
        hidden (list):
            List of (layer(int), datatype(int)) tuples that are hidden by default in
            the visualization of the integrated circuit layout.
        
        depth (int):
            Depth of visualization of the integrated circuit layout, defining
            if the cells referenced by the main cells of the library are shown
            or not.
        
        colour (dict):
            Dictionary defining the colour of each (layer,datatype) tuple
        
        pattern (dict):
            Dictionary defining the filling pattern of each (layer,datatype) tuple
        
        backAnnotation (bool):
            Boolean flag signaling if the user wants to perform
            back annotation of extracted values or not
            
        backAnnotationMap (dict):
            A dictionary generated from BackannotationDict class that
            allows for the mapping of a physical quantity value to a 
            given hex code of a colour
        
        polygonPhysicalQuantityMap (dict):
            a dictionary for mapping each cell's polygon to a physical quantity value,
            allowing to perform back annotation of the resistance values
        
        backGround (str):
            Hex code of the background colour of the layout visualizer of Speedster
        
        width (int):
            Width of the layout visualizer frame
        
        height (int):
            Height of the layout visualizer frame
    """
    def __init__(
        self,
        library:GdsLibrary = None,
        cells: dict = None,
        hidden = [],
        depth = 0,
        colour = {},
        pattern = {},
        backAnnotation = False,
        backAnnotationMap = {},
        polygonPhysicalQuantityMap = {},
        backGround = "#FFFFFF", # default is white background
        width = 900,
        height = 650,
    ):
        """_summary_
        Initializes the SpeedsterLayoutViewer class
        Args:
            library (Library, optional): . Defaults to None.
        """
        pass