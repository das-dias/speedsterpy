from loguru import logger
import os
from spdstrlib import (
    __workspace_filename__,
    __workspace_lib_path__,
    SpdstrWorkspaceLib,
    SpdstrWorkspace,
    read, 
    write, 
    dump, 
    load,
    __version__,
    getParent, 
)

def test_version():
    assert __version__ == '0.1.34'

def test_workspace_lib_creation():
    lib = SpdstrWorkspaceLib(
        libPath=__workspace_lib_path__,
        fileName=__workspace_filename__,
    )
    assert type(lib) == SpdstrWorkspaceLib
    assert lib.libPath == __workspace_lib_path__
    assert lib.libFileName == __workspace_filename__
    #save workspace library
    dump(lib)

def test_workspace_creation():
    """_summary_
    Test the creation of a workspace
    """ 
    #load library from standard path:
    path = os.path.join(__workspace_lib_path__, __workspace_filename__)
    lib = load(path)
    
    #parent = os.path.dirname(os.path.abspath(__file__))
    file = os.path.abspath(__file__)
    parent = getParent(file, 1)
    logger.info("\nParent directory:{}".format(parent))
    newWorkspace = SpdstrWorkspace("test_project")
    projectDir = "{}/resources/temp".format(parent)
    newWorkspace.saveWorkspaceDir(projectDir)
    newWorkspace.saveTechFile("{}/test.tlef".format(projectDir))
    newWorkspace.saveLayoutFile("{}/test.gds".format(projectDir))
    lib.add(newWorkspace)
    print(str(newWorkspace))
    assert type(newWorkspace) == SpdstrWorkspace
    assert newWorkspace.name == "test_project"
    assert newWorkspace.workspacePath == "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrlib/resources/temp"
    assert newWorkspace.techPath == "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrlib/resources/temp/test.tlef"
    assert newWorkspace.layoutPath == "/Users/dasdias/Documents/SoftwareProjects/speedsterpy/spdstrlib/resources/temp/test.gds"
    logger.info("Success creating a new project:\n{}".format(str(newWorkspace)))
    #create a testbench directory
    newWorkspace.createTestbench()
    # create a testbench output child directory
    newWorkspace.createTestbenchOutputDir()
    # create a testbench configuration .toml file
    newWorkspace.createTestbenchCfgFile()
    
    print(str(newWorkspace))
    
    # save the workspace library
    dump(lib)
    
def test_workspace_write():
    """_summary_
    Test the writing of a workspace to json file
    """
    # load the workspace library
    path = os.path.join(__workspace_lib_path__, __workspace_filename__)
    lib = load(path)
    
    #parent = os.path.dirname(os.path.abspath(__file__))
    file = os.path.abspath(__file__)
    parent = getParent(file, 1)
    logger.info("\nParent directory:{}".format(parent))
    newWorkspace = SpdstrWorkspace("test_project_v2")
    projectDir = "{}/resources/temp2".format(parent)
    newWorkspace.saveWorkspaceDir(projectDir)
    newWorkspace.saveTechFile("{}/test2.tlef".format(projectDir))
    newWorkspace.saveLayoutFile("{}/test2.gds".format(projectDir))
    lib.add(newWorkspace)
    write(newWorkspace)
    print("Updated library:")
    try:
        print(str(lib))
    except Exception as e:
        pass
    
    # save the workspace library
    dump(lib)
    

def test_workspace_read():
    """_summary_
    Test the reading of a workspace from json file
    """
    # load the workspace library
    path = os.path.join(__workspace_lib_path__, __workspace_filename__)
    lib = load(path)
    wspath = lib["test_project_v2"]["fullpath"]
    newWorkspace2 = read(wspath)
    logger.info("\n{}".format(str(newWorkspace2)))
    

    
