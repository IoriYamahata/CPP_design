#! /usr/bin/env python3



import argparse
import re
import sys



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log'                    ,
                        type     = str             ,
                        help     = 'input log file',
                        required = True            ,)

    return parser


def read_log(log):
    with open(log) as f:
        pattern = re.compile('   Created a bond between .:(\d+) and .:(\d+); changed residue names to "CYX "')
        bonds   = []
        for line in f:
            line = line.rstrip('\r\n')
            m = pattern.fullmatch(line)
            if m:
                resnum_1 = int(m.group(1))
                resnum_2 = int(m.group(2))
                bonds.append((resnum_1, resnum_2))

    return bonds


def print_leapin(bonds):
    print('source leaprc.protein.ff14SB')
    print('source leaprc.water.tip3p')
    print('loadAmberParams frcmod.ionsjc_tip3p')
    print('')
    print('mol = loadPDB in.pdb')
    print('')
    for bond in bonds:
        print('bond mol.%d.SG mol.%d.SG' % bond)
    print('')
    print('addIons2 mol Na+ 0')
    print('addIons2 mol Cl- 0')
    print('')
    print('solvateBox mol TIP3PBOX 12.0')
    print('')
    print('saveAmberParm mol leap.parm7 leap.rst7')
    print('savePDB mol leap.pdb')
    print('quit')


def main():
    parser = get_parser()
    args = parser.parse_args()

    bonds = read_log(args.log)
    print_leapin(bonds)
    return 0



if __name__ == '__main__':
    sys.exit(main())
