#! /usr/bin/env python3



import argparse
import sys



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pdb'                    ,
                        type     = str             ,
                        help     = 'input PDB file',
                        required = True            ,)

    return parser


def read_pdb(pdb):
    with open(pdb) as f:
        lines = []
        for line in f:
            line = line.rstrip('\r\n')
            lines.append(line)

    return lines


def serialize(lines):
    chain_prev   = ''
    resid_prev   = ''
    resid_serial = -1
    lines_serial = []
    for line in lines:
        if line[: 4] == 'ATOM' or line[: 3] == 'TER':
            chain = line[21     ]
            resid = line[22 : 27]
            if   chain != chain_prev:
                chain_prev    = chain
                resid_prev    = resid
                resid_serial += 3
            elif resid != resid_prev:
                chain_prev    = chain
                resid_prev    = resid
                resid_serial += 1

            line = line[: 22] + f'{resid_serial : >4d} ' + line[27 :]

        lines_serial.append(line)

    return lines_serial


def main():
    parser = get_parser()
    args = parser.parse_args()

    lines = read_pdb(args.pdb)
    lines = serialize(lines)
    print('\n'.join(lines))
    return 0



if __name__ == '__main__':
    sys.exit(main())
