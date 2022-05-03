"""
data.py : provides additional 
data structures and interfaces 
that are essential for the good
interaction between the tool 
and the user

[author]    Diogo André Silvares Dias
[date]      2022-04-17
[contact]   das.dias@campus.fct.unl.pt
"""
from enum import Enum
from collections import defaultdict
from itertools import starmap

class Unimplemented(object):
    """_summary_
    Unimplemented type raise class
    Args:
        object (Unimplemented): raise unimplementation error
    """
    def __init__(self, filepath):
        raise TypeError ("{} : Module not implemented yet".format(filepath))
    

class GdsLayerPurpose(Enum):
    """_summary_
    Enumerator for the purpose
    of a GDS layer
    Args:
    """
    DRAWING     = "drawing"
    TEXT        = "text"
    IDENTIFIER  = "identifier"
    DUMMY       = "dummy"
    GATE        = "gate"
    RESISTOR    = "resistor"
    MODEL       = "model"
    HIGH_VOLTAGE= "high voltage"
    FUSE        = "fuse"
    BLOCK       = "block"
    TERMINAL1   = "terminal1"
    TERMINAL2   = "terminal2"
    TERMINAL3   = "terminal3"
    BOUNDARY    = "boundary"
    LABEL       = "label"
    NET         = "net"
    PIN         = "pin"
    CUT         = "cut"
    PROBE       = "probe"
    SHORT       = "short"
    MASK        = "mask"
    MASK_ADD    = "mask add"
    MASK_DROP   = "mask drop"
    WAFFLE_DROP = "waffle drop"
    BACKANNOTATION = "backannotation"
    HIGHLIGHTING   = "highlighting"
    
class GdsTable(dict):
    """_summary_
    Standard GDSII table data structure dictionary
    representation, using the (layer number, datatype)
    as keys and the layer name, layer purpose and description
    as values
    """
    __slots__ = [
        "table"
    ]
    def __init__(self):
        self.table = {}
    
    def __dict__(self) -> dict:
        """_summary_
        Return the GDSTable data structure as a dict
        """
        ret = {}
        for key, value in self.table.items():
            ret[key] = {
                "name": value["name"],
                "purpose": [pur.name for pur in value["purpose"]],
                "description": value["description"]
            }
        return ret

    def __str__(self) -> str:
        """_summary_
        Return the GDSTable data structure as a string
        """
        ret  = "-----------------\n"
        ret += "GDSII Table\n"
        for key, value in self.table.items():
            ret += "-----------------\n"
            ret += "Layer: {}\n".format(key)
            ret += "Name: {}\n".format(value["name"])
            ret += "Purpose: {}\n".format([val.name for val in value["purpose"]])
            ret += "Description: {}\n".format(value["description"])
        ret += "-----------------"
        return ret
    
    def __getitem__(self, key: tuple) -> object:
        return self.table[key]

    def __setitem__(self, key: tuple, value: dict) -> None:
        self.table[key] = value
    
    def __delitem__(self, key: tuple) -> None:
        del self.table[key]
    
    def __contains__(self, key: tuple) -> bool:
        return key in self.table.keys()

    def __iter__(self) -> iter:
        return iter(self.table.keys())
    
    def items(self) -> iter:
        return self.table.items()
    
    def keys(self) -> iter:
        return self.table.keys()
    
    def values(self) -> iter:
        return self.table.values()
    
    def purposes(self) -> iter:
        return self.table.values()["purpose"]
    
    def names(self) -> list:
        return self.table.values()["name"]
    
    def descriptions(self) -> list:
        return self.table.values()["description"]
    
    def add(
        self,
        layer: int,
        dataType: int,
        name: str,
        purpose: list,
        description: str,
    ):
        self.table[(layer, dataType)] = {
            "name": name,
            "purpose": [GdsLayerPurpose[pur] for pur in purpose],
            "description": description
        }
    
    def parseData(self, yamlDict: dict) -> None:
        """_summary_
        Auto-generate the GDSTable data structure
        from a given yaml recovered dict
        Args:
            yamlDict (dict): a dictionary containing the GDSTable
                             data structure
        """
        for key, value in yamlDict.items():
            if not "name" in value.keys():
                raise TypeError("The parsed yamlDict must contain the \"name\" key")
            if not "purpose" in value.keys():
                raise TypeError("The parsed yamlDict must contain the \"purpose\" key")
            if not "description" in value.keys():
                raise TypeError("The parsed yamlDict must contain the \"description\" key")
            self.add(
                key[0],
                key[1],
                value["name"],
                value["purpose"],
                value["description"]
            )
            
    def getLayerName(self, layer: int, dataType: int) -> str:
        """_summary_
        Return the layer name associated to the given layer number
        and data type
        Args:
            layer (int): the layer number
            dataType (int): the data type
        """
        return self.table[(layer, dataType)]["name"]
    
    def getLayerPurpose(self, layer: int, dataType: int) -> list:
        """_summary_
        Return the layer purpose associated to the given layer number
        and data type
        Args:
            layer (int): the layer number
            dataType (int): the data type
        """
        return self.table[(layer, dataType)]["purpose"]
    # TODO: ADD FILTER FUNCTIONS
    def getGdsTableEntriesFromPurpose(self, purpose: GdsLayerPurpose):
        """_summary_
        Get a new GdsTable object from a GdsTable object
        by selecting only the rows with the specified purpose
        Args:
            other   (GdsTable)          : GdsTable object to be filtered
            purpose (GdsLayerPurpose)   : purpose to be aapplied as filter
        Returns:
            GdsTable: filtered GdsTable object
        """
        newTab = GdsTable()
        for key, value in self.items():
            if purpose in value["purpose"]:
                newTab.table[key] = value
        return newTab if len(newTab.table) > 0 else None

    def getGdsTableEntriesFromLayerName(self, layerName: str):
        """_summary_
        Get a new GdsTable object from a GdsTable object
        by selecting only the rows with the specified layer name
        Args:
            other   (GdsTable)          : GdsTable object to be filtered
            layerName (str)             : layer name to be aapplied as filter
        Returns:
            GdsTable: filtered GdsTable object
        """
        newTab = GdsTable()
        for key, value in self.items():
            if layerName in value["name"]:
                newTab.table[key] = value
        return newTab if len(newTab.table) > 0 else None

    def getGdsLayerDatatypeFromLayerNamePurpose(self, layerName: str, purpose: GdsLayerPurpose) -> list:
        """_summary_
        Get the layer,datatype of a layer from a GdsTable object
        by selecting only the rows with the specified layer name and purpose
        Args:
            other       (GdsTable)  : GdsTable object to be filtered
            layerName   (str)       : layer name to be aapplied as filter
            purpose     (str)       : purpose to be aapplied as filter
        Returns:
            list[tuple]: list of layer,datatype tuples for each layer
        """
        ret:list = []
        for key, value in self.items():
            if layerName == value["name"] and purpose in value["purpose"]:
                ret.append(key)
        return ret if len(ret) > 0 else None
    
    def getDrawingMetalLayersMap(self, maxMetalNum = 15) -> dict:
        """_summary_
        Get a dictionary with the metal layers used for drawing
        metal shapes and interconnects in the imported gds table.
        Args:
            table       (GdsTable)      : GDS table associated with a given tech node.
            maxMetalNum (int, optional) : Maximum number of routing metal layers. Defaults to 15.
        Returns:
            dict: "layer name" : (layer, datatype) map
        """
        # get the ordered metal layers and 
        via = "via"
        met = "met"
        layerMap = {}
        # don't use the first layer, since it is a via layer
        #layerMap["mcon"] = getGdsLayerDatatypeFromLayerNamePurpose("mcon", GdsLayerPurpose.DRAWING)
        layerMap["met1"] = self.getGdsLayerDatatypeFromLayerNamePurpose("met1", GdsLayerPurpose.DRAWING)[0]
        layerMap["via"] = self.getGdsLayerDatatypeFromLayerNamePurpose( "via", GdsLayerPurpose.DRAWING)[0]
        for i in range(2,maxMetalNum):#maximum number of metal layers 
            aux = self.getGdsLayerDatatypeFromLayerNamePurpose("{}{}".format(met,str(i)), GdsLayerPurpose.DRAWING)
            if aux != None:
                layerMap["{}{}".format(met,str(i))] = aux[0]
                lastVia = i
            else: # if the routing metal layer is not found, stop
                break
            aux = self.getGdsLayerDatatypeFromLayerNamePurpose("{}{}".format(via,str(i)), GdsLayerPurpose.DRAWING)
            if aux != None:
                layerMap["{}{}".format(via,str(i))] = aux[0]
                lastMetal = i
        return layerMap
    
    def addBackannotation(self):
        """_summary_
        Add dedicated drawing layers for performing backannotation
        on the layout
        Args:
            table (GdsTable): the imported table to which the items will be added
        Returns:
            GdsTable: the same library as the one received, but with the added items
        """
        # check if the backannotation layer is already present
        purposeTable = self.getGdsTableEntriesFromPurpose(GdsLayerPurpose.BACKANNOTATION)
        if purposeTable != None:
            raise Exception("Backannotation layer already present")
        
        # back annotation will only be performed on metal layers or vias layers
        metalLayerNames = list(self.getDrawingMetalLayersMap().keys())
        forbiddenLayerDatatypes = defaultdict(list) # list of forbidden (layer, datatype) tuples
        possibleLayerDatatypes = {} # list of possible (layer, datatype) tuples
        for name in metalLayerNames:
            for key in self.getGdsTableEntriesFromLayerName(name).keys():
                if key not in forbiddenLayerDatatypes[name]:
                    forbiddenLayerDatatypes[name].append(key) 
            for key in forbiddenLayerDatatypes[name]:
                if (key[0], key[1]+1) not in forbiddenLayerDatatypes[name]:
                    possibleLayerDatatypes[name] = (key[0], key[1]+1)
        for name, key in possibleLayerDatatypes.items():
            self.add(key[0], key[1], name, [GdsLayerPurpose.BACKANNOTATION.name], "{} backannotation".format(name))
        return self

    def addHighlight(self):
        """_summary_
        Add dedicated a layer dedicated to
        highlighting the selected net
        Args:
            table (GdsTable): the imported table to which the items will be added
        Returns:
            GdsTable: the same library as the one received, but with the added items
        """
        # check if the backannotation layer is already present
        highlightingTable = self.getGdsTableEntriesFromPurpose(GdsLayerPurpose.HIGHLIGHTING)
        if highlightingTable != None:
            raise Exception("Highlighting layer already present")
        
        for layer, datatype in self.keys():
            if (layer+1, 0) not in self.keys():
                self.add(layer+1, 0, "highlight", [GdsLayerPurpose.HIGHLIGHTING.name], "net highlighting")
                break # break out of loop
        return self
