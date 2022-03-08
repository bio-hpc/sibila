from Common.Input.InputParams import InputParams
from datetime import datetime

args = InputParams().read_params()

file_dataset = args.dataset
options = args.option

args.folder = '{}_{}'.format(args.folder, datetime.now().strftime("%Y-%m-%d"))
print(args.folder)
