# TransE+LINE
This is an implementation of the TransE+LINE model.
The model considers both the triplets and text data. For the triplets, it uses the objective function of TransE for optimiztion; for the text data, it first constructs a co-occurrence matrix between words, and uses the objective function of LINE for training. The two objective functions are jointly optimized.

The codes rely on two external packages (Eigen and GSL). After installing the packages, users need to change the package paths in the makefile. Then we can compile the code and use the running script run.sh to train.

Step 1: Constructing word co-occurrence matrix using data2w.cpp
Options:
-text : text data
-output-ww : output word co-occurrence matrix
-output-words : output word vocabulary
-window : window size for construction
-min-count : word min count for construction

Step 2: Training embedding
Options:
-entity : entity+word vocabulary file, which consists of N lines, where N is the total number of entities and words. Each line contains an entity name or word.
-relation : relation vocabulary file, which consists of R lines, where R is the number of relations. Each line contains a relation name.
-network : co-occurrence matrix
-triple : training triplet file. Each line describes a triplet, with the format <Head> <Tail> <Relation>
-output-en : output entity embedding file
-output-rl : output relation embedding file
-binary : whether to output embeddings in the binary format
-size : embedding dimension
-samples : number of training samples (in million), 300 is a good default.
-alpha : learning rate. 0.01 is a good default.
-threads : number of threads for training
