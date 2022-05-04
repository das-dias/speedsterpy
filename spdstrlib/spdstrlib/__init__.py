__version__ = '0.1.34'
__description__ = "Speedster project library management tool."
__date__ = "2022-04-12"
__author__ = "Diogo André Silvares Dias"
__annotations__ = ""

import argparse
import sys
from loguru import logger
import pickle
import os
from .read import (
    load,
    read
)
from .write import(
    dump,
    write
)
from .util import *
from .data import *

#create the global workspace lib path save
__workspace_lib_path__ = f"{getParent(os.getcwd(), 0)}/wslib"
if not os.path.exists(__workspace_lib_path__):
    os.makedirs(__workspace_lib_path__)
__workspace_filename__ = "wslib.bin"

def verboseInfo():
    print(f"Version      : {__version__} ({__date__})")
    print(f"Authors      : {__author__}")
    print(f"Description  : {__description__}")

__arg_to_func__ = {
    SelectedOp.CREATE   : handleCreation,
    SelectedOp.DELETE   : handleDeletion,
    SelectedOp.SHOW     : handleShow,
    SelectedOp.LIST     : handleList,
}

def run(subparser, *args, **kwargs) -> None:
    logger.info("Speedster$\nLibrary Manager : {}".format(__file__))
    argv = None
    try:
        argv = subparser.parse_args(sys.argv[2:])
    except Exception as e:
        logger.error(f"{e.__class__.__name__} - {e}")
    if argv.info:
        verboseInfo()
        return None
    # load the workspace library
    libPath = os.path.join(__workspace_lib_path__, __workspace_filename__)
    lib = None
    try:
        lib = load(libPath)
    except FileNotFoundError:
        logger.info("Workspace library not found. Creating a new one.")
        # create new library on default path
        lib = SpdstrWorkspaceLib(
            libPath= __workspace_lib_path__,
            fileName= __workspace_filename__
        )
        # dump the library on the standard path
        dump(lib)
    #handle mutually exclusive arguments
    try:
        args = handleMutuallyExclusive(argv)
        for arg in args:
            __arg_to_func__[arg](argv, lib)
    except Exception as e:
        logger.error(f"{e.__class__.__name__} - {e}")
    # dump the library on the standard path
    dump(lib)
    