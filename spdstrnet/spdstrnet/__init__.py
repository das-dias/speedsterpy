__version__ = '0.1.0'
__name__ = "spdstrnet"
__description__ = "Speedster net tracing, extraction and highlighting functionalities."
__date__ = "2022-04-16"
__author__ = "Diogo André Silvares Dias"
__annotations__ = ""

from .geometry import *
from .net import *

def verboseInfo():
    print(f"Version      : {__version__} ({__date__})")
    print(f"Authors      : {__author__}")
    print(f"Description  : {__description__}") 