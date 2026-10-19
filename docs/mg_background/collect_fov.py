#!/usr/bin/env python3
# Collect the raw numbers of the figures into docs/mg_background/data/ (run once; the figures
# script reads only these files).
import os, re
HOME = os.path.expanduser('~')
D = HOME + '/RutticeQCD/docs/mg_background/data'
os.makedirs(D, exist_ok=True)

pseudo = re.compile(r'pseudo z = ([+-][\d.]+) ([+-][\d.]+)i: sigma_min\(z - A~ M\) = (\S+)')
lam = re.compile(r'arnoldi\s+lambda_\d+ = ([+-]\S+) ([+-]\S+)i')
far = re.compile(r'lambda = ([+-]\S+) ([+-]\S+)i')
fov = re.compile(r'A~ M : Re of the field of values in \[(\S+), (\S+)\]')
for setup, files in {
    'S12': ['pseudo_12_stout.txt', 'pseudo3_12_stout.txt', 'dense_12_stout.txt'],
    'B12': ['pseudo_12_m18.txt', 'pseudo3_12_m18.txt', 'dense_12_m18.txt', 'dense_12_m18_b.txt'],
}.items():
    with open(f'{D}/pseudo_{setup}.tsv', 'w') as out:
        out.write('re\tim\tsigma_min\tsource\n')
        for f in files:
            p = f'{HOME}/fov/{f}'
            if not os.path.exists(p):
                continue
            for line in open(p):
                m = pseudo.search(line)
                if m:
                    out.write(f'{m.group(1)}\t{m.group(2)}\t{m.group(3)}\t{f}\n')
for setup, f in {'S12': 'far_12_stout.txt', 'B12': 'far_12_m18.txt'}.items():
    text = open(f'{HOME}/fov/{f}').read()
    with open(f'{D}/spectrum_{setup}.tsv', 'w') as out:
        out.write('re\tim\tkind\n')
        for m in lam.finditer(text):
            out.write(f'{m.group(1)}\t{m.group(2)}\tsmallest_re\n')
        for line in text.splitlines():
            if 'largest |lambda -' in line:
                m = far.search(line)
                out.write(f'{m.group(1)}\t{m.group(2)}\tfarthest_from_1\n')
        m = fov.search(text)
        out.write(f'{m.group(1)}\t0\tfov_min\n{m.group(2)}\t0\tfov_max\n')
print(open(f'{D}/spectrum_S12.tsv').read())
print(sum(1 for _ in open(f'{D}/pseudo_S12.tsv')), sum(1 for _ in open(f'{D}/pseudo_B12.tsv')))
