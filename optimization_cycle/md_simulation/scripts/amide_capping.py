#! /usr/bin/env python3



import argparse
import sys





def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pdb'                           ,
                        type     = str                    ,
                        help     = 'input preped PDB file',
                        required = True                   ,)

    return parser



def read_pdb(pdb):
    with open(pdb) as f:
        lines = []
        for line in f:
            line = line.rstrip('\r\n')
            lines.append(line)

    return lines




def edit_binder_N_terminal(lines):
    edited_lines = []
    for index, line in enumerate(lines):
        if line.startswith("ATOM") and "N   NMA B" in line:
            edited_lines.append(line.replace("N   NMA B", "N   NHE B"))
        else:
            edited_lines.append(line)
    return edited_lines




def remove_binder_NME_capping(lines):
    removed_lines = []
    for line in lines:
        if "NMA B" in line:
            continue
        else:
            removed_lines.append(line)
    return removed_lines





def main():
    parser = get_parser()
    args   = parser.parse_args()

    lines  = read_pdb(args.pdb)
    lines  = edit_binder_N_terminal(lines)
    lines  = remove_binder_NME_capping(lines)
    print('\n'.join(lines))
    return 0





if __name__ == '__main__':
    sys.exit(main())