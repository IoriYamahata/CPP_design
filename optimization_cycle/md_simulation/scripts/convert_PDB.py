#! /usr/bin/env python3



import argparse
import csv
import sys



CONVERSIONS_ALL = [ (17, 21,      'NMA ',      'NME '),
                    (17, 21,      'HIS ',      'HID '),
                    (12, 21, ' CA  NME ', ' C   NME '), ]

# CONVERSIONS_AA     = [ ('ASP ', 'ASH '),
#                        ('GLU ', 'GLH '),
#                        ('ARG ', 'ARN '),
#                        ('LYS ', 'LYN '),
#                        ('HID ', 'HIE '),
#                        ('HID ', 'HIP '),
#                        ('CYS ', 'CYX '), ]
CONVERSIONS_AA = [ ('HID ', 'HIE '),
                   ('HID ', 'HIP '),
                   ('CYS ', 'CYX '), ]



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pdb'                    ,
                        type     = str             ,
                        help     = 'input PDB file',
                        required = True            ,)

    parser.add_argument('--mae'                    ,
                        type     = str             ,
                        help     = 'input MAE file',
                        required = True            ,)

    return parser


def read_pdb(pdb):
    with open(pdb) as f:
        lines = []
        for line in f:
            line = line.rstrip('\r\n')
            lines.append(line)

    return lines


def read_mae(mae):
    with open(mae) as f:
        condition2conversion = {}
        data = csv.reader(f              ,
                          delimiter = ' ',
                          quotechar = '"',)
        for line in data:
            if len(line) <= 12:
                continue

            resnum = int(line[ 7])
            resch  =     line[ 8]
            chain  =     line[10]
            res    =     line[12]
            for source, target in CONVERSIONS_AA:
                if res == target:
                    condition = (21, 27, chain + ('%4d' % resnum) + resch)
                    conversion = (17, 21, source, target)
                    if condition in condition2conversion:
                        assert conversion == condition2conversion[condition]
                    else:
                        condition2conversion[condition] = conversion

    return condition2conversion


def is_H(line):
    return line[: 4] == 'ATOM' and line[-3 :] == 'H  '


def remove_H(lines):
    lines_new = [ line for line in lines if not is_H(line) ]
    return lines_new


def is_conect(line):
    return line[: 6] == 'CONECT'


def remove_conect(lines):
    lines_new = [ line for line in lines if not is_conect(line) ]
    return lines_new


def find_internal_caps_and_external_residues(lines, caps):
    removes = []
    is_external = False
    chain_prev  = ''
    resid_prev  = ''
    for line in lines:
        if line[: 4] == 'ATOM':
            resname = line[17 : 21]
            chain   = line[21     ]
            resid   = line[22 : 27]
            if chain == chain_prev and resid == resid_prev:
                continue

            if chain != chain_prev:
                is_external = True

            if resname in caps: 
                if is_external:
                    is_external = False
                else:
                    internal_cap = resname + chain + resid
                    removes.append(internal_cap)
            else:
                if is_external:
                    external_res = resname + chain + resid
                    removes.append(external_res)

            chain_prev  = chain
            resid_prev  = resid

    return removes


def is_internal_caps_or_external_residues(line, removes):
    return (line[: 4] == 'ATOM' or line[: 3] == 'TER') and line[17 : 27] in removes


def remove_internal_caps_and_external_residues(lines):    
    
    removes_ACE = find_internal_caps_and_external_residues(         lines , ['ACE ']          )
    removes_NMA = find_internal_caps_and_external_residues(reversed(lines), ['NMA ', 'NHE '])
    removes = removes_ACE + removes_NMA
    lines_new = [ line for line in lines if not is_internal_caps_or_external_residues(line, removes) ]
    return lines_new


def convert_all_line(line):
    if line[: 4] == 'ATOM' or line[: 3] == 'TER':
        for begin, end, source, target in CONVERSIONS_ALL:
            if line[begin : end] == source:
                line = line[: begin] + target + line[end :]

    return line


def convert_all(lines):
    lines_new = [ convert_all_line(line) for line in lines ]
    return lines_new


def convert_line(line, condition2conversion):
    if line[: 4] == 'ATOM' or line[: 3] == 'TER':
        for (begin_cond, end_cond, str_cond), (begin_conv, end_conv, source_conv, target_conv) in condition2conversion.items():
            if line[begin_cond : end_cond] == str_cond:
                assert line[begin_conv : end_conv] == source_conv
                line = line[: begin_conv] + target_conv + line[end_conv :]

    return line


def convert(lines, condition2conversion):
    lines_new = [ convert_line(line, condition2conversion) for line in lines ]
    return lines_new


def main():
    parser = get_parser()
    args = parser.parse_args()

    lines                = read_pdb(args.pdb)
    condition2conversion = read_mae(args.mae)

    lines = remove_H(lines)
    lines = remove_conect(lines)
    lines = remove_internal_caps_and_external_residues(lines)
    lines = convert_all(lines)
    lines = convert(lines, condition2conversion)
    print('\n'.join(lines))
    return 0



if __name__ == '__main__':
    sys.exit(main())
