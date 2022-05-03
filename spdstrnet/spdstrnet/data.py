"""_summary_
data.py contains the main data structures and
classes for the performance of the resistance
extraction method leveraged by image processing

[author]    Diogo André Silvares Dias
[date]      2022-04-17
[contact]   das.dias@campus.fct.unl.pt
"""
from enum import Enum
import numpy as np
from spdstrutil import Unimplemented
class SpeedsterPortType(Enum):
    """_summary_
    SpeedsterPointType enum
    Args:
        None
        INPUT : input point
        OUTPUT : output point
        IO: input and output point
    """
    None
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class SpeedsterPort(object):
    __slots__ = [
        "name",
        "ioType",
        "layer",
        "purpose",
        "resistance",
        "location",
        "width",
    ]
    def __init__(
        self, 
        name: str = "new_port",
        ioType: SpeedsterPortType = None,
        resistance: float = 0.0,
        location: list = (0,0),
        width: float = 1.0,
        layer: str = "met1",
        purpose: str = "probe",
    ):
        self.name = name
        self.ioType = ioType
        self.layer = layer
        self.purpose = purpose
        self.resistance = resistance
        self.location = location
        self.width = width
        
    def __str__(self):
        return "Port: {} Type: {} Location: {} in {} Resistance: {}".format(
            self.name,
            self.ioType.name,
            self.location,
            self.layer,
            self.resistance,
        )
    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "ioType": self.ioType.name,
            "resistance": self.resistance,
            "location": self.location,
            "width": self.width,
            "layer": self.layer,
            "purpose": self.purpose,
        }
    def parseData(self, yamlDict: dict):
        """_summary_
        Parses the data of a port
        Args:
            yamlDict (dict): a dictionary containing the port data, resulting from a yaml file
            
        """
        if not "name" in yamlDict.keys():
            raise KeyError("Port must have a name")
        self.name = yamlDict['name']
        if not "ioType" in yamlDict.keys():
            raise KeyError("Port must have an I/O Type")
        self.ioType = SpeedsterPortType[yamlDict['ioType']]
        if not "resistance" in yamlDict.keys():
            raise KeyError("Port must have a resistance")
        self.resistance = yamlDict['resistance']
        if not "location" in yamlDict.keys():
            raise KeyError("Port must have a location")
        self.location = yamlDict['location']
        if not "width" in yamlDict.keys():
            raise KeyError("Port must have a width")
        self.width = yamlDict['width']
        if not "layer" in yamlDict.keys():
            raise KeyError("Port must have a layer")
        self.layer = yamlDict['layer']
        if not "purpose" in yamlDict.keys():
            raise KeyError("Port must have a purpose")
        self.purpose = yamlDict['purpose']
    
    def get_polygon(self):
        x = self.location[0]
        y = self.location[1]
        return [
            (x - self.width/2, y - self.width/2),
            (x + self.width/2, y - self.width/2),
            (x + self.width/2, y + self.width/2),
            (x - self.width/2, y + self.width/2),
        ]
        
class SpeedsterPortLibrary(object):
    def __init__(self):
        self.ports = {}
    def __str__(self):
        ret = "-----------------\n"
        ret += "Port Library\n"
        for value in self.ports.values():
            ret += "-----------------\n"
            ret += "{}\n".format(value)
        ret += "-----------------\n"
        return ret
    
    def __iter__(self):
        return iter(self.ports.values())
    
    def add(self, port: SpeedsterPort):
        """_summary_
        Adds a port to the library
        Args:
            port (SpeedsterPort): the port to be added
        Returns:
            SpeedsterPortLibrary: the library with the new port
        """
        if port.name in self.ports.keys():
            raise KeyError("Port {} already exists".format(port.name))
        self.ports[port.name] = port
        return self
        
    def remove(self, portName):
        """_summary_
        Remnoves a port from the library
        Args:
            portName (_type_): _description_
        Returns:
            SpeedsterPortLibrary: the library without the port
        Raises:
            KeyError: _description_
        """
        if portName not in self.ports.keys():
            raise KeyError("Port {} does not exist".format(portName))
        del self.ports[portName]
        return self

    def __dict__(self) -> dict:
        """_summary_
        Returns a dictionary with the ports
        Returns:
            dict: _description_
        """
        ret = {}
        for key, value in self.ports.items():
            ret[key] = value.__dict__()
        return ret

    def parseData(self, yamlDict: dict):
        """_summary_
        Parses the data of a port
        Args:
            yamlDict (dict): a dictionary containing the port data, resulting from a yaml file
            
        """
        for portName, port in yamlDict.items():
            newPort = SpeedsterPort()
            newPort.parseData(port)
            self.add(newPort)
    
    
class SpeedsterResMap(object):
    """_summary_
    Standard resistance map data structure
    for the Speedster tool, to map resistance values
    to each fragment of a PolygonSet
    """
    __slots__ = [
        "r"
    ]
    
    def __init__(self, resistances = []):
        if resistances != []:
            if type(resistances[0]) != float:
                raise TypeError("The resistances must be a list of floats")
        self.r = np.array(resistances)
    
    def __dict__(self) -> dict:
        return {"r": list(self.r)}
    
    def __eq__(self, other) -> bool:
        """_summary_
        Returns if the Speedster ResMap geometry data structure is equal to another
        Args:
            other (object): _description_
        """
        if type(other) != SpeedsterResMap:
            return False
        return np.array_equal(self.r, other.r)
    
    def __ne__(self, other) -> bool:
        """_summary_
        Returns if the Speedster ResMap geometry data structure is not equal to another
        Args:
            other (object): _description_
        """
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        return str(self.r)
    
    def __repr__(self) -> str:
        return str(self.r)
    
    def __len__(self) -> int:
        return len(self.r)
    
    def __getitem__(self, index: int) -> float:
        return self.r[index]
    
    def __iter__(self) -> iter:
        return iter(self.r)

    def __setitem__(self, index: int, value: float) -> None:
        self.r[index] = value
        
    def __delitem__(self, index: int) -> None:
        del self.r[index]
    
    def __append__(self, value: float) -> np.array:
        self.r = np.append(self.r, value)
        return self.r
    
    def parse_data(self, yamlDict: dict) -> None:
        """_summary_
        Auto-generate the Speedster ResMap data structure
        from a given yaml recovered dict
        Args:
            yamlDict (dict): a dictionary containing the Speedster ResMap
                             data structure
        """
        if not "r" in yamlDict:
            raise TypeError("The parsed yamlDict must contain the \"r\" key")
        self.r = np.array( yamlDict["r"] )

# TODO : Develop a SpeedsterLayoutResistanceMap to save the
# resistance map of a GdsCell representing a net

# TODO : Develop a SpeedsterConnectionGraph to save
# connections between polygons of the same and of different layers
# treating polygons and vias as nodes and the connections as edges

# TODO : Develop a SpeedsterChargeMobilityGraph
# to save the generated graphs for current path in the layout




# TODO : Develop a SpeedsterInterLayerChargeMobilityGraph
# to save the generated graphs for the mobility 
# of charge between successive metal layers of the
# integrated circuit
        
# TODO : develop an extraction engine, that 
# controls the extraction flux
# and provides textual results to console