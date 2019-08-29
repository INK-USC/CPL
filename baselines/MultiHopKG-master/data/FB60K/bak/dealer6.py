from tqdm import tqdm

with open("bak/train.txt","r") as tn:
    output = open("train.triples","w")
    lin = tn.readlines()
    for line in tqdm(lin):
        lines = line.split()
        output.write(lines[0] + "\t" + lines[2] + "\t" + lines[1] + "\n")

with open("bak/test.txt","r") as tn:
    output = open("test.triples","w")
    lin = tn.readlines()
    for line in tqdm(lin):
        lines = line.split()
        output.write(lines[0] + "\t" + lines[2] + "\t" + lines[1] + "\n")

with open("bak/dev.txt","r") as tn:
    output = open("dev.triples","w")
    lin = tn.readlines()
    for line in tqdm(lin):
        lines = line.split()
        output.write(lines[0] + "\t" + lines[2] + "\t" + lines[1] + "\n")

