import re
import sys


with open(sys.argv[1], 'r') as f:
    lines = [line for line in f]


with open(sys.argv[2], 'w') as f:
    for line in lines:
        triple = line.split()
        if triple[2] == "/media_common/netflix_genre/titles":
            f.write(line)
