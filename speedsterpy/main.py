import argparse
from loguru import logger
import os
import sys
from spdstrlib import run as librun
from spdstrres import run as rpexrun

from util import (
    platformInfo,
    appInfo,
    setupArgParser,
)

SYSTOKENS = [
    ("lib", "library",  "create a new project library library associated with a given technology"),
    ("rpex","resistance-extraction", "parasitic resistance extraction"),
    #("pow", "power", "current density and dissipated power extaction"),
    #("cpex","capacitance-extraction", "parasitic capacitance extraction")
]

SYSARGS = {
    "lib": [
        ("-l", "list all available workspaces", "<>", None),
        ("-c", "create new workspace", '<name>', str),
        ("-s", "show workspace info", '<name>', str),
        ("-d", "delete workspace", '<name>', str),
        ("-ws",   "workspace directory", '<dirpath>', str), # to save images and other output files
        ("-tlef", "technology LEF file path", '<filepath>', str),
        ("-gds",  "GDS file path", '<filepath>', str),
        ("-tab",  "gds table file path", '<filepath>', str),
        ("-lef",  "LEF file path with pad and port locations", '<filepath>', str),
        ("-net",  "circuit (SPICE) netlists file path to extract netlist info", '<filepath>', str),
        ("-tb",   "testbench directory path", '<dirpath>', str),
        ("-o",    "library extraction output directory", '<dirpath>', str),
    ],
    
    "rpex": [
        ("-ptp",    "perform point-to-point resistance extraction, producing a .spr file", '<>', None),
        ("-v",      "visualize the extracted point to point resistance", '<>', None),
        ("-ws",     "project workspace name to run the extraction on", '<filepath>', str),
        ("-b",      "produce a .toml file with the consumed time and memory computing resources benchmarks of the extraction", '<>', None),
        ("-spef",   "produce a .spef file with a given spefFileName", '<filepath>', str),
        ("-net",    "generate a netlist file with a given netName from extracted data", '<filepath>', str),
        ("-o",      "produces .yaml files with the resulting structures from parasitic extraction", '<>', None),
    ],
}


SYSFUNCS = {
    'lib': librun,
    'rpex': rpexrun,
}

def main():
    sysPath = sys.argv[0]
    logger.info(f"Speedster$ Parent Dir : {sysPath}")
    #logger.info("Dir: {}".format(os.path.abspath(__file__))) 
    parser = setupArgParser(SYSTOKENS, SYSARGS, SYSFUNCS)
    if len(sys.argv) <= 1: # append "help" if no arguments are given
        sys.argv.append("-h")
    elif sys.argv[1] in [tok[0] for tok in SYSTOKENS]:
        if len(sys.argv) == 2:
            sys.argv.append("-h")   # append help when only 
                                    # one positional argument is given
    try:
        args = parser.parse_args()
        if args.cmd: # if command of the original parser was given
            args.cmd()
        else:
            args = parser.parse_args(sys.argv[1:])
            try:
                args.func()
            except Exception as e:
                logger.error(e)
    except argparse.ArgumentError as e: # catching unknown arguments
        logger.warning(e)
    except Exception as e:
        logger.error(e)
    
    
    
if __name__ == "__main__":
    main()