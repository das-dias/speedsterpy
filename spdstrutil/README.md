# spdstrutil : Speedster utilities package

This sub-package contains useful datastructures and functions that allow for a greater versatility of the operation of the general Speedster application, when it comes to its application to multiple semiconductors technology nodes.
Its main features are, for now:
- A GDSII table datastructure
- Useful table accessing methods for quick selection and filtering of GDSII layers
- A GDSII table parser implemented using ```CSV(.csv)``` and ```YAML(.yaml)``` files read and write functions
---
## Examples (with code snippets)
Reading/importing an existing ```GDSII``` table from a ```.csv(CSV)``` file associated with a given semiconductor technology node:
```python
import spdstrutil as util

path = "../foo/tech/gds_table.csv"
gds_table = util.readGdsTable(path)
```
Writing/exporting an imported ```GDSII``` table to disk as ```.yaml(YAML)``` file format:
```python
write_path = "../bar/tech/"
#only .yaml file writing is supported in the current version
util.writeGdsTable(gds_table, write_path)
# the file will automatically be save in path: "../bar/tech/gds_table.yaml"
```
Reading/importing a newly exported ```GDSII``` table from a ```.yaml(YAML)``` file, associated with a given semiconductor technology node:
```python
read_yaml_path = "../foo/tech/gds_table.yaml"
new_gds_table = util.readGdsTable(read_yaml_path)
```
Get only the table entries associated with the ```GDSII``` layers featuring the purpose of Drawing:
```python
drawing_table = new_gds_table.getGdsTableEntriesFromPurpose(util.GdsLayerPurpose.DRAWING)
assert type(drawing_table) == util.GdsTable
```
Get only the table entries associated with the ```GDSII``` layer featuring a given layer name:
```python
met4_table = new_gds_table.getGdsTableEntriesFromLayerName("met4")
assert type(met4_table) == util.GdsTable
```
Obtaing a ```"layer name" : (layer, datatype)``` map (dictionary) containing a mapping solemnly for the Drawing metal layers (Interconnect Routing and VIAs) that are present in the imported ```GDSII``` table:
```python
#getDrawingMetalLayersMap selects metal layers up until "met14" (15 metal layers) by default. An higher number of maximum metal layers can be specified as input to getDrawingMetalLayersMap.
metal_layer_map = new_gds_table.getDrawingMetalLayersMap()
```

This utilities package also includes a ```@timer``` wrapper that allows to log into console how much time (in microseconds) it took for a function/ method to run:
```python
from spdstrutil import timer
@timer
def wait_ms(msec: float) -> str:
    import time
    time.sleep(msec*1e-3)
    return f"Waited {msec} milli seconds"
print(wait_ms(10))
```
Console Output:
```
.........2022-05-03 23:58:12.201 | INFO     | spdstrutil.util:wrapper:30 - Function: wait_ms	Runtime: 12568.000 μs.
Waited 10.0 milli seconds
```