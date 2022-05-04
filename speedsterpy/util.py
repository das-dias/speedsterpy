import sys # to absorve system arguments
import argparse # to build a command line interface using a general purpose parser
import platform # get platform and machine info
from functools import wraps # wrap function parse inputs from subparser arguments 
#local
from __init__ import(
    __version__, 
    __author__, 
    __email__,
    __description__,
    __annotations__,
    __date__
) 

def platformInfo() -> None:
    """_summary_
    Prints platform info to console
    """
    ret =  f"Python      : {str(sys.version.split('\n'))}\n"
    ret += f"System      : {str(platform.system())}\n"
    ret += f"Machine     : {str(platform.machine())}\n"
    ret += f"Platform    : {str(platform.platform())}\n"
    ret += f"Version     : {str(platform.version())}\n"
    #return ret
    print(ret)

def appInfo() -> None:
    """_summary_
    Prints app info to console
    """
    ret =  f"Speedster, Version {__version__} ({__date__})\n"
    ret += f"Author      : {__author__}\n"
    ret += f"Email       : {__email__}\n"
    ret += f"Annotations : {__annotations__}\n"
    #return ret
    print(ret)

def mapSubparserToFun(func, subparser):
    """_summary_
    Maps subparser to callback function
    Args:
        func (Function): callback function
        subparser (_type_): subparser object
    Returns:
        result (_type_): result of the callback function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(subparser, *args, **kwargs)
    return wrapper

def setupArgParser(
    subSysTokens, 
    subSysArgs,
    funcs
) -> argparse.ArgumentParser:
    """_summary_
    Setup the argparser Parser object to develop
    the command line interface of the main app
    Args:
        subSysTokens (_type_): _description_
        subSysArgs (_type_): _description_
        funcs (_type_): _description_
    Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser( description=f"{__description__}", exit_on_error= False)
    # mutually exclusive arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q", "--quiet", 
        action="store_true", 
        help="quiet verbose"
    )
    group.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="print verbose"
    )
    # general arguments
    parser.add_argument(
        "-a", 
        "--app",
        help = "show app info", 
        action = 'store_const',
        const = appInfo,
        dest = 'cmd'
    )
    parser.add_argument(
        "-p", 
        "--platform", 
        help = "show platform info", 
        action = "store_const",
        const = platformInfo,
        dest = 'cmd'
    )
    # setup subparsers
    subparsers = parser.add_subparsers()
    for cmd, extCmd, hel in subSysTokens:
        subparser = subparsers.add_parser(cmd, help=hel)
        subparser.set_defaults(
            func=mapSubparserToFun(funcs[cmd], subparser)
        )
        if cmd in subSysArgs.keys():
            for item in subSysArgs[cmd]:
                arg = item[0]
                hel = item[1]
                metavar = item[2]
                typo = item[3]
                if typo != None:
                    subparser.add_argument(
                        arg, 
                        nargs=1, 
                        help=hel,
                        metavar=metavar if metavar != '<>' else '',
                        type = typo
                    )
                else:
                    subparser.add_argument(
                        arg, 
                        action = 'store_true', 
                        help=hel,
                    )
            # add info option
            subparser.add_argument("-i", "--info", action="store_true", help="show tool info")      
    return parser

