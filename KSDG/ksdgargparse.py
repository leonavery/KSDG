import sys
import shlex
from argparse import ArgumentParser, SUPPRESS

class KSDGParser(ArgumentParser):
    """An argument parser for KSDG scripts

    KSDGParser is used like argparse.ArgumentParser -- you create one,
    call the add_argument method to add arguments to it, then call the
    parse_args methdos to parse the command line. However, it also
    extracts FEniCS and PETSc commandline arguments. The syntax for
    passing these is:

    program --fenics fenicsarg1 fenicsarg2 ...
    program --petsc petscarg1 petscarg2 ...
    program myarg1 ... --fenics fenicsarg1 fenicsarg2 ... -- myarg2 myarg3
    program myarg1 ... --fenics fenicsarg1 fenicsarg2 ... --
            --petsc petscarg1 petscarg2 ... -- myarg2 myarg3

    parse_args returns a Namespace object that contains the results of
    parsing your object in the usual way. In addition, it will contain
    two names, 'fenics' and 'petsc', whose values are lists of strs
    that can be passed to petsc4py.init and fenics.parameters.parse in
    order to set defaults for those subsystems.
    """

    subsystems = ['fenics', 'petsc']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, fromfile_prefix_chars='@',
                         allow_abbrev=False, **kwargs)
        #
        # The following are just for the sake of the help
        # message. '--fenics' and '--petsc' will be stripped before
        # arguments are passed to the parser, so these options will
        # not usually be activated. 
        #
        self.add_argument('--fenics', action='append', default=SUPPRESS,
                          help='FEniCS subsystem arguments: \
                                terminate with --, \
                                --fenics --help for help')
        self.add_argument('--petsc', action='append', default=SUPPRESS,
                          help='PETSc subsystem arguments: \
                                terminate with --, \
                                --petsc -help for help')

    def convert_arg_line_to_args(self, arg_line, comment_char='#'):
        """Override the function in argparse to handle comments"""
        return shlex.split(arg_line, comments=True)
        cpos = arg_line.find(comment_char)
        if cpos >= 0:
            arg = arg_line[:cpos].strip()
        else:
            arg = arg_line.strip()
        if arg:
            return([arg])
        else:
            return([])
    
    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        #
        # ***** warning *****
        # Here I call a private method of ArgumentParser to handle
        # file indirection.
        #
        args = self._read_args_from_files(args)
        sargs = [[], []]
        for s, subsystem in enumerate(self.subsystems):
            while('--' + subsystem in args):
                f = args.index('--' + subsystem)
                try:
                    e = args.index('--', f + 1)
                except ValueError:
                    e = len(args)
                sargs[s] += args[f+1:e]
                args[f:e+1] = []
        ns = super().parse_args(args)
        for s, subsystem in enumerate(self.subsystems):
            setattr(ns, subsystem, sargs[s])
        return ns
        

def main():
    parser1 = KSDGParser('KSDG Parser', fromfile_prefix_chars='@')
    parser1.add_argument('-1', '--opt1', action='store_true')
    parser1.add_argument('-2', '--opt2')
    print(parser1.parse_args(
        ['-1', '--fenics', '--linearsolver=gmres', '--', '-2', 'filename']
    ))
    print(parser1.parse_args(
        ['--fenics', '--linearsolver=gmres', '--', '--petsc', '-help',
         '--', '-2', 'filename']
    ))
    print(parser1.parse_args())

if __name__ == '__main__':
    main()
    
