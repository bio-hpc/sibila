import argparse
from os.path import splitext, isfile, basename
from Tools.IOData import IOData
from Tools.DataNormalization import DataNormalization
from Models.Utils.CrossValidation import CrossValidation
from Tools.DatasetBalanced import DatasetBalanced
""" 
  this class is responsible for receiving json files with parameters of the option and if any is different from the
   default, will be added
"""


class InputParams:
    ALLOW_EXTENSIONS_DATASET = ['csv', 'pkl']

    def __init__(self):
        self.iodata = IOData()

    @staticmethod
    def file_choices(parser, choices, file_name):
        ext = splitext(file_name)[1][1:]
        if ext not in choices:
            parser.error("\nfile doesn't end with one of {}\n".format(choices))
        if not isfile(file_name):
            parser.error("\nfile doesn't exits {}\n".format(file_name))
        return file_name

    def check_params(self, args):
        if args.model is not None and any(v is not None
                                          for v in [args.option, args.parameters, args.balanced, args.crossvalidation]):
            self.iodata.print_e('ERROR: -m argument is not compatible with -o, -t, -p, -b and -cv')

    def read_params(self):
        """
            Reads the parameters using argparse
        """
        print("")
        options = self.iodata.read_all_options()
        parser = argparse.ArgumentParser(description='SIBILA', add_help=True)
        parser.add_argument('-d',
                            '--dataset',
                            help="Database file",
                            type=lambda s: self.file_choices(parser, self.ALLOW_EXTENSIONS_DATASET, s),
                            required=True)
        parser.add_argument('-o',
                            '--option',
                            nargs='+',
                            choices=options + ["ALL"],
                            help='Type of model',
                            type=str.upper)
        parser.add_argument(
            '-n',
            '--normalize',
            nargs='+',
            choices=list(DataNormalization.METHODS.keys()),
            help='Normalize datasaet (mm = minmax, ma = MaxAbsScaler)',
            type=str.upper,
        )
        parser.add_argument('-q', '--queue', help="launches the interpretability methods as job", action='store_true')

        parser.add_argument('-p', '--parameters', help="File parameters json", nargs='+', type=argparse.FileType('r'))
        parser.add_argument('-t',
                            '--trainsize',
                            required=False,
                            type=int,
                            choices=range(50, 91),
                            metavar="[50-90]",
                            help='Part of the dataset for training',
                            default=80)
        parser.add_argument('-s', "--seed", help="Random state", type=int, default=2020)
        parser.add_argument('-f', "--folder", help="Folder out", type=str)
        parser.add_argument('-cv',
                            '--crossvalidation',
                            help='Cross validation',
                            type=str.upper,
                            choices=list(CrossValidation.METHODS.keys()))
        parser.add_argument('-m', '--model', help='Model(s) to predict with', type=str, nargs='+')
        parser.add_argument('-r', '--regression', help='Regression', action='store_true')

        parser.add_argument('-b',
                            '--balanced',
                            nargs='+',
                            help='Balanced dataset',
                            type=str.upper,
                            choices=list(DatasetBalanced.METHODS.keys()))
        
        args = parser.parse_args()
        self.check_params(args)
        if args.model:
            args.option = None
            args.trainsize = None
        else:
            args.option = options if (args.option[0] == "ALL") else args.option
        if args.parameters:
            [
                self.iodata.print_e("Parameter files must be json") for i in args.parameters
                if splitext(i.name)[1] != ".json"
            ]
            args.parameters = [i.name for i in args.parameters]
        args.introduced_folder = True
        if not args.folder:
            args.folder = splitext(basename(args.dataset))[0]
            args.introduced_folder = False

        return args
