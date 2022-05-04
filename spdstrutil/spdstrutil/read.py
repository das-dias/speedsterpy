"""
read.py : provides functionalities
to import saved data structures in files into
the program memory
"""
import yaml
import csv
import pickle
import os
from .data import(
    GdsLayerPurpose,
    GdsTable,
)

def readGdsTable(filePath = "") -> GdsTable:
    """_summary_
    Reads the gds table
    Args:
        fileName (str, optional): _description_. Defaults to "".

    Raises:
        ValueError: _description_
    """
    
    if filePath == "":
        raise ValueError("No file name provided")
    
    path = os.path.abspath(filePath)
    if not os.path.exists(path):
        raise ValueError(f"Invalid file path provided : {path}")
    
    head, tail = os.path.split(path)
    filename, extension = os.path.splitext(tail)
    if extension not in [".csv", ".yaml"]:
        raise ValueError("Invalid format provided: accepts \"json\" or \"csv\"")
    
    # declare a new empty gds table
    gdsTable = GdsTable()
    if extension == ".csv":
        header = {
                "Layer name"        : 0,
                "Purpose"           : 1,
                "GDS layer:datatype": 2,
                "Description"       : 3
            }
        with open(path, "r") as csvFile:
            file = csv.reader(csvFile)
            #establish the heard order
            for line in file:
                for i,identifier in enumerate(line, 0):
                    header[identifier] = i
                break
            for line in file:
                # ignore empy lines
                if line[ header["GDS layer:datatype"] ] != "":
                    # preprocessing of the lines
                    #preprocess purpose line
                    splits = line[ header["Purpose"] ].split(',')
                    purposes = []
                    for spl in splits:
                        if spl != "":
                            key = spl[1:] if spl[0] == " " else spl
                            #print(";"+key+";")
                            purposes.append(GdsLayerPurpose(key).name)
                    #print(purposes)
                    #preprocess layer:datatype line
                    splits = line[ header["GDS layer:datatype"] ].split(':')
                    layer = int(splits[0])
                    datatype = int(splits[1])
                    gdsTable.add(
                        layer=layer,
                        dataType=datatype,
                        name=line[ header["Layer name"] ],
                        purpose=purposes,
                        description=line[ header["Description"] ].replace('\n', '')
                    )
    else:# extension == ".yaml":
        with open(path, "r") as yamlFile:
            gdsTableDict = yaml.load(yamlFile, Loader=yaml.FullLoader)
        gdsTable.parseData(gdsTableDict)
    return gdsTable if len(gdsTable.table) > 0 else None