from argparse import Namespace
from loguru import logger
from enum import Enum
import os
from .data import *
from .read import *
from .write import *

def getParent(path, levels = 1):
    """_summary_
    Get the parent directory of a path
    according to the specified level of
    depth in the tree
    Args:
        path    (str)   : child path of the parent directory
        levels  (int)   : number of levels to go up in the tree
    """
    common = path
    for _ in range(levels+1):
        common = os.path.dirname(os.path.abspath(common))
    return common

class SelectedOp(Enum):
    """_summary_
    Selected operation enumeration
    for easy access
    Args:
        CREATE : create a new workspace
        DELETE : delete a workspace
        SHOW   : show a workspace
        LIST   : list all workspaces
    """
    None
    CREATE  = 1
    DELETE  = 2
    SHOW    = 3
    LIST    = 4

def handleMutuallyExclusive(argv: Namespace) -> list:
    """_summary_
    handler for the mutually exclusive options
    of the command line
    Args:
        namespace (Namespace):  namespace object
                                holding the parsed arguments
                                from console, using
                                argparse subparser
    """
    ret = []
    if (
        (argv.s or argv.d or argv.l) and 
        ( argv.c or argv.tab or argv.tlef or argv.lef or argv.gds or argv.tb or argv.o or argv.net)
    ): 
        raise ValueError("Show, List and Delete operations are mutually exclusive with other operations")
    if argv.c:
        ret.append(SelectedOp.CREATE)
    if argv.s:
        ret.append(SelectedOp.SHOW)
    if argv.d:
        ret.append(SelectedOp.DELETE)
    if argv.l:
        ret.append(SelectedOp.LIST)
    return ret
    
def handleCreation(argv: Namespace, lib: SpdstrWorkspaceLib) -> None:
    """_summary_
    handler for the creation of a new workspace
    Args:
        namespace (Namespace):  namespace object
                                holding the parsed arguments
                                from console, using
                                argparse subparser
    """
    # mandatory values to be parsed from console
    if not argv.c:
        raise ValueError("No workspace name provided")
    
    workspaceName = argv.c[0]
    if not argv.ws:
        raise ValueError("No workspace directory was provided")
    
    workspaceDir = argv.ws[0]
    if not argv.tlef:
        raise ValueError("No .tlef tech rule file was provided")
    
    techFilePath = argv.tlef[0]
    if not argv.gds:
        raise ValueError("No .gds layout file was provided")
    
    layoutFilePath = argv.gds[0]
    if not argv.lef:
        raise ValueError("No .lef ports and pins file was provided")
    
    lefFilePath = argv.lef[0]
    #optional values to be parsed from console
    netlistFilePath = argv.nl[0] if argv.net else None
    testbenchDir = argv.tb[0] if argv.tb else None
    outputDir = argv.o[0] if argv.o else None
    tableFilePath = argv.tab[0] if argv.tab else None
        
    # after collecting all the values, create the workspace
    args = {
        "name"                  : workspaceName,
        "workspacePath"         : workspaceDir,
        "techPath"              : techFilePath,
        "layoutPath"            : layoutFilePath,
        "portsPath"             : lefFilePath,
    }
    newWorkspace = SpdstrWorkspace(selfDict=args)
    if netlistFilePath:
        newWorkspace.saveNetlistFile(netlistFilePath)
    if testbenchDir:
        newWorkspace.saveTestbenchDir(testbenchDir)
    else:
        # if no testbench directory was provided,
        newWorkspace.createTestbench() # create a default testbench
        newWorkspace.createTestbenchCfgFile() # create a default testbench configuration file
    if outputDir:
        newWorkspace.saveTestbenchOutput(outputDir)
    else: # if no output directory was provided,
        newWorkspace.createTestbenchOutputDir() # create a default testbench output directory
    if tableFilePath:
        newWorkspace.saveGdsTableFile(tableFilePath)
    # write new workspace to memory
    write(newWorkspace)
    # save new workspace to library
    lib.add(newWorkspace, overWrite=True)
    logger.info("Workspace \"{}\" created successfully".format(workspaceName))
    return newWorkspace

def handleDeletion(argv: Namespace, lib: SpdstrWorkspaceLib) -> SpdstrWorkspace:
    """_summary_
    handler for the deletion of a workspace
    Args:
        namespace (Namespace):  namespace object
                                holding the parsed arguments
                                from console, using
                                argparse subparser
    """
    if not argv.d:
        raise ValueError("No workspace name provided")
    workspaceName = argv.d[0]
    #fetch the workspace from the library
    workspace = read(lib[workspaceName]["fullpath"])
    # erase the workspace from the library
    lib.remove(workspaceName)
    #return the deleted workspace
    logger.info(f"Workspace \"{workspaceName}\" deleted successfully")
    return workspace

def handleShow(argv: Namespace, lib : SpdstrWorkspaceLib) -> None:
    """_summary_
    handler for the show of a workspace
    Args:
        namespace (Namespace):  namespace object
                                holding the parsed arguments
                                from console, using
                                argparse subparser
    """
    if not argv.s:
        raise ValueError("No workspace name provided")
    workspaceName = argv.s[0]
    #fetch the workspace from the library
    workspace = read(lib[workspaceName]["fullpath"])
    #print the workspace
    logger.info(f"\n{workspace}")

def handleList(argv: Namespace, lib: SpdstrWorkspaceLib) -> None:
    """_summary_
    handler for the listing of workspaces
    available in the workspace library
    Args:
        namespace (Namespace):  namespace object
                                holding the parsed arguments
                                from console, using
                                argparse subparser
    """
    logger.info(f"\n{str(lib)}")