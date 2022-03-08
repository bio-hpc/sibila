import argparse
import random
import pandas as pd
import os

MULTI_VALUES = [
    "M_", "WHO grade", "Stage_", "tumor growth pattern", "tumor_budding_general_", "mucine_", "necrosis_", "PI3K_exon10"
    "PI3K_exon20", "KRAS_", "MSI_status_", "CIMP_Weisemberg_", "CIMP_Ogino_", "CAGNA1G", "CDKN2A", "CRABP1", "IGF2_",
    "MLH1_", "NEUROG1", "RUNX3", "SOCS1"
]
DELIMITER = ';'


def check_multi_value(col):

    for i in MULTI_VALUES:
        if col.startswith(i):
            return i
    return ""


def coun_multi_value_check(tag_col):
    cnt_cols = []
    for i in lst_columns:
        if i.startswith(tag_col):
            cnt_cols.append(i)
    return cnt_cols


def get_random_min_max(col, name_col):
    max = col.max()
    min = col.min()
    if min == max:
        return min
    if int(min) == min and int(max) == max:  #integuer
        return random.randrange(min, max + 1)
    else:
        return random.uniform(min, max)


parser = argparse.ArgumentParser(description='SIBILA', add_help=True)
parser.add_argument('dataset', help="Database file")
parser.add_argument('-m', '--max_items', type=int, help="Database file", default=1)
args = parser.parse_args()
data = pd.read_csv(args.dataset)
lst_columns = list(data.columns)
name_out = '{}_random.csv'.format(os.path.splitext(args.dataset)[0])
df = pd.DataFrame(columns=lst_columns)
dct_row = {}
lst_cols = []

for row in range(args.max_items):
    # print("")
    for name_col in lst_columns:
        tag_col = check_multi_value(name_col)
        if name_col in lst_cols:
            pass
        elif tag_col != "" and name_col not in lst_cols:
            lst_cols = coun_multi_value_check(tag_col)
            random_col = random.randrange(0, len(lst_cols))
            s = ""
            for i in range(len(lst_cols)):
                if i == random_col:
                    s += lst_cols[i] + " 1 "
                    dct_row[lst_cols[i]] = 1
                else:
                    s += lst_cols[i] + " 0 "
                    dct_row[lst_cols[i]] = 0
            # print('{}  {}     {}'.format(tag_col, random_col, s))
        else:
            value = get_random_min_max(data[name_col], name_col)
            # print('{}  {} '.format(name_col, value))
            dct_row[name_col] = value
    for k, v in dct_row.items():
        df.loc[row, k] = v
# print(df)
df.to_csv(name_out, decimal='.')
