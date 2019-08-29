import operator
import math, random, sys, csv 
from utils import parse, print_results
from tqdm import tqdm
from multiprocessing import Process, Pool, Queue
import time




class PageRank:
    def __init__(self, graph, directed):
        self.graph = graph
        self.V = len(self.graph)
        self.d = 0.85
        self.directed = directed
        self.ranks = dict()

    def rank(self):
        def job(directed, range, d, v, graph, ranks, res):
            def job_directed(graph, n, ranks):
                rs = 0
                outlinks = len(list(graph.out_edges(n[1])))
                if outlinks > 0:
                    rs += (1 / float(outlinks)) * ranks[n[1]]
                return rs

            def job_undirected(graph, n, ranks):
                rs = 0
                if ranks[n] is not None:
                    outlinks = len(list(graph.neighbors(n)))
                    rs += (1 / float(outlinks)) * ranks[n]
                return rs

            tempranks = {}
            for key, node in tqdm(range):
                rank_sum = 0
                curr_rank = node.get('rank')

                if directed:
                    neighbors = graph.out_edges(key)
                    for n in neighbors:
                        rank_sum += job_directed(graph, n, ranks)
                else:
                    neighbors = graph[key]
                    for n in neighbors:
                        rank_sum += job_undirected(graph, n, ranks)

                # actual page rank compution
                tempranks[key] = ((1 - float(d)) * (1 / float(v))) + d * rank_sum
            res.put(tempranks)

        thr = 16
        for key, node in tqdm(self.graph.nodes(data=True)):
            if self.directed:
                self.ranks[key] = 1/float(self.V)
            else:
                self.ranks[key] = node.get('rank')

        for i in range(10):
            now = time.time()
            print("epoch",i)
            print(len(self.ranks))
            datarange = list(self.graph.nodes(data=True))
            siz = len(datarange)
            mark = siz // thr
            sets = [datarange[mark*i:mark*(i+1)] for i in range(thr)]
            sets[15] = datarange[mark*15:]
            jobs = []
            ranksnew = Queue()

            for mm in range(thr):
                print("starting process",mm)
                proc = Process(target=job,
                               args=(self.directed,sets[mm],self.d,self.V,self.graph,self.ranks,ranksnew))
                proc.start()
                jobs.append(proc)

            newranks = dict()
            collected = 0
            while collected < thr:
                while not ranksnew.empty():
                    result = ranksnew.get()
                    collected += 1
                    newranks.update(result)
            #for job in jobs: job.join()
            #ranksnew.join_thread()
            self.ranks = newranks
            nownow = time.time()
            print("epoch", i, " , time elapsed:", nownow - now)

        return p

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Expected input format: python pageRank.py <data_filename> <directed OR undirected> <save path>')
    else:
        filename = sys.argv[1]
        isDirected = False
        if sys.argv[2] == 'directed':
            isDirected = True

        graph = parse(filename, isDirected)
        p = PageRank(graph, isDirected)
        p.rank()

        sorted_r = sorted(iter(p.ranks.items()), key=operator.itemgetter(1), reverse=True)

        filen = open(sys.argv[3],"w")
        for tup in sorted_r:
            print('{0:30} :{1:10}'.format(str(tup[0]), tup[1]), file=filen)
            print('{0:30} :{1:10}'.format(str(tup[0]), tup[1]))

 #       for node in graph.nodes():
 #          print node + rank(graph, node)

            #neighbs = graph.neighbors(node)
            #print node + " " + str(neighbs)
            #print random.uniform(0,1)

def rank(graph, node):
    #V
    nodes = graph.nodes()
    #|V|
    nodes_sz = len(nodes) 
    #I
    neighbs = graph.neighbors(node)
    #d
    rand_jmp = random.uniform(0, 1)

    ranks = []
    ranks.append( (1/nodes_sz) )
    
    for n in nodes:
        rank = (1-rand_jmp) * (1/nodes_sz) 
        trank = 0
        for nei in neighbs:
            trank += (1/len(neighbs)) * ranks[len(ranks)-1]
        rank = rank + (d * trank)
        ranks.append(rank)

 
