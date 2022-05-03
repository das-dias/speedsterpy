__version__ = '0.1.1'
__name__ = "spdstrutil"
__description__ = "Speedster utilities package."
__date__ = "2022-04-19"
__author__ = "Diogo André Silvares Dias"
__annotations__ = ""

from .data import(
    Unimplemented,
    GdsLayerPurpose,
    GdsTable,
)
from .read import (
    readGdsTable,
)
from .write import (
    writeGdsTable,
)
from .util import *

def verboseInfo():
    print("Version      : {} ({})".format(__version__, __date__))
    print("Authors      : {}".format(__author__))
    print("Description  : {}".format(__description__))
