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
    REGRESSION_MODELS = ['ANN', 'KNN', 'RF', 'DT', 'SVM', 'XGBOOST', 'LR', 'BAG', 'VOT']


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

    def check_params(self, args, parser):
        if args.dataset is None and args.explanation is None:
            parser.error("{-d, --dataset} is compulsory when {-e, --explantion} is not present")
        elif args.explanation is not None:
            return

        if args.model is not None and any(v is not None
                                          for v in [args.option, args.parameters, args.balanced, args.crossvalidation]):
            self.iodata.print_e('-m argument is not compatible with -o, -t, -p, -b and -cv')
        
        if args.model is None:
            opt_aux = [value for value in args.option if value in self.REGRESSION_MODELS]
            if args.regression == True:
                if len(args.option) == 1 and (args.option[0] != 'ALL' and args.option[0] not in self.REGRESSION_MODELS):
                    self.iodata.print_e('-r argument is only valid with -r [ALL, {}] parameter'.format(', '.join(self.REGRESSION_MODELS)))
                elif len(args.option) > 1 and len(args.option) != len(opt_aux):
                    self.iodata.print_e('-r argument is only valid with -r [ALL, {}] parameter'.format(', '.join(self.REGRESSION_MODELS)))

    def read_params(self):
        """
            Reads the parameters using argparse
        """
        print("")
        options = self.iodata.read_all_options()
        
        # Mover 'VOT' al final de la lista
        options = sorted(options, key=lambda x: (x == "VOT", x))

        options_reg = [value for value in options if value in self.REGRESSION_MODELS and value != 'ANN']
        parser = argparse.ArgumentParser(description='SIBILA', add_help=True)
        parser.add_argument('-d',
                            '--dataset',
                            help="Dataset file in CSV or PKL format",
                            type=lambda s: self.file_choices(parser, self.ALLOW_EXTENSIONS_DATASET, s)
                            )
        parser.add_argument('-o',
                            '--option',
                            nargs='+',
                            choices=options + ["ALL", "VOT"],
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
        parser.add_argument('--skip-dataset-analysis', help='Skip dataset analysis plots', action='store_true', default=False)
        parser.add_argument('--skip-interpretability', help='Do not compute interpretability on test data', action='store_true', default=False)
        parser.add_argument('-e', '--explanation', help='Explain a dataset given a .pkl file', type=str)

        args = parser.parse_args()
        self.check_params(args, parser)
        
        if args.model and not args.explanation:
            args.option = None
            args.trainsize = None
        elif not args.explanation:
            
            if "VOT" in args.option:
                # Dynamically include all base models for VOT and ensure VOT is executed last
                base_models = [model for model in options if model != "VOT"]
                print(f"VOT activated: using base models {base_models}")
                #args.option = base_models + ["VOT"]  # Re-add VOT to execute it after base models
                args.option = ["VOT"]

            elif not args.regression:
                args.option = options if (args.option[0] == "ALL") else args.option
            else:
                args.option = options_reg if (args.option[0] == "ALL") else args.option
        if args.parameters:
            [
                self.iodata.print_e("Parameter files must be json") for i in args.parameters
                if splitext(i.name)[1] != ".json"
            ]
            args.parameters = [i.name for i in args.parameters]
        args.introduced_folder = True
        if not args.folder and not args.explanation:
            args.folder = splitext(basename(args.dataset))[0]
            args.introduced_folder = False

        return args
