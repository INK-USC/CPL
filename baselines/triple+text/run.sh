#!/bin/sh

# g++ data2w.cpp -o data2w

#./data2w -text /data/toni/TransE+LINE-UMLS/text.txt \
#         -output-ww network-umls.txt -output-words entity-umls.txt -window 5 -min-count 10

for siz in 10 20 40 70 100; do
    ./embed -entity entity-umls.txt \
            -relation /data/toni/TransE+LINE-UMLS/relations.txt \
            -network network-umls.txt \
            -triple "/data/toni/TransE+LINE-UMLS/"$siz"text/train.txt" \
            -output-en ety_emb/entity-$siz-umls.emb \
            -output-rl rlt_emb/relation-$siz-umls.emb \
            -binary 1 -size 50 -negative 5 -samples 100 -threads 45 -alpha 0.01
done