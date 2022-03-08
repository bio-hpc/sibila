import configparser


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


TXT_ERRORS = "Tests/Errors.txt"
FORMAT_ERROR = bcolors.FAIL + '\n\n\t ERROR: {} -- {}' + bcolors.ENDC
FORMAT_ERR_TXT = '{} ' + bcolors.WARNING + ' {} ' + bcolors.ENDC


def get_error(error):

    config = configparser.RawConfigParser()
    config.read(TXT_ERRORS)
    details_dict = dict(config.items('SECTION_ERROR'))

    return FORMAT_ERROR.format(error, details_dict[error.lower()])


def get_error_txt(error, txt):
    return FORMAT_ERR_TXT.format(get_error(error), txt)
