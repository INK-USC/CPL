#!/usr/bin/env bash

#siz=benchmarks/yagoD
#echo "$siz"
#echo "$siz" >> results.txt
#echo "TransE:" >> results.txt
#python3 session_TransE.py $siz
#echo "DistMult:" >> results.txt
#python3 session_DistMult.py $siz
#echo "ComplEx:" >> results.txt
#python3 session_ComplEx.py $siz
#echo "RESCAL:" >> results.txt
#python3 session_RESCAL.py $siz
#
siz=benchmarks/autonomous
echo "$siz"
echo "$siz" >> results.txt
echo "DistMult:" >> results.txt
python3 session_DistMult.py $siz

siz=benchmarks/math
echo "$siz"
echo "$siz" >> results.txt
echo "DistMult:" >> results.txt
python3 session_DistMult.py $siz

siz=benchmarks/reddit
echo "$siz"
echo "$siz" >> results.txt
echo "DistMult:" >> results.txt
python3 session_DistMult.py $siz

#siz=benchmarks/yago
#echo "$siz"
#echo "$siz" >> results.txt
#echo "TransE:" >> results.txt
#python3 session_TransE.py $siz
#echo "DistMult:" >> results.txt
#python3 session_DistMult.py $siz
#echo "ComplEx:" >> results.txt
#python3 session_ComplEx.py $siz
#echo "RESCAL:" >> results.txt
#python3 session_RESCAL.py $siz
#
#siz=benchmarks/wiki
#echo "$siz"
#echo "$siz" >> results.txt
#echo "TransE:" >> results.txt
#python3 session_TransE.py $siz
#echo "DistMult:" >> results.txt
#python3 session_DistMult.py $siz
#echo "ComplEx:" >> results.txt
#python3 session_ComplEx.py $siz
#echo "RESCAL:" >> results.txt
#python3 session_RESCAL.py $siz