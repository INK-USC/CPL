# CPL: Collaborative Policy Learning

TensorFlow implementation of EMNLP 2019 paper ***[Collaborative Policy Learning for Open Knowledge Graph Reasoning](https://arxiv.org/abs/1909.00230)***. Moving to PyTorch.

Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Citation](#citation)
* [Building and Running](#building-and-running)
     * [Prerequisites](#prerequisites)
     * [Running](#running)
     * [Parameters](#parameters)
     * [Baselines](#baselines)
* [Performance](#performance)
     * [Datasets](#datasets)
     * [Results](#results)
* [File Structure](#file-structure)
* [TODO](#todo)
<!--te-->

### TR;DL
  - Sparsity issue in knowledge graph reasoning can benefit from dynamically enrich the graph structure based on a background text corpus 
  - We show that a collaborative policy learning method (path-finding agent & fact extraction agent) can seek most relevant facts to augment search space of path-finding agent and improve the KG reasoning performance.
  - We release two datasets of general and biomedical domains for this new task.

## Introduction

We study the new task of Open Knowledge Graph Reasoning (OKGR), where the new facts extracted from the text corpora will be used to augment the graph dynamically while performing reasoning.
All the recent joint graph and text embedding methods focus on learning better knowledge graph embeddings for reasoning, but we consider adding more facts to the graph from the text to improve the reasoning performance and further provide interpretability. 

 However, most facts so extracted may be noisy or irrelevant to the path inference process. Moreover, adding a large number of edges to the graph will create an ineffective search space and cause scalability issues to the path finding models. So we need to dynamically add edges as we walk through the graph.
 
![Moltivation-gif](https://github.com/shanzhenren/CPL/blob/master/emnlp-gif.gif?raw=true)

### Method Overview

To address the above challenges for OKGR, we propose our **Collaborative Policy Learning** (CPL) framework to jointly train two RL agents in a mutually enhancing manner. 
In CPL, besides training a **reasoning** agent for inference path finding, we further introduce a **fact extracting** agent, which learns the policy to select relevant facts extracted from the corpus, based on the context of the reasoning process and the corpus.

![image-20181020190951048](https://github.com/shanzhenren/GraphPath/blob/master/README.assets/image-20181020190951048.png)


## Cite this paper

```Bibtex
@article{fu2019collaborative,
  title={Collaborative Policy Learning for Open Knowledge Graph Reasoning},
  author={Fu, Cong and Chen, Tong and Qu, Meng and Jin, Woojeong and Ren, Xiang},
  journal={EMNLP},
  year={2019}
}
```

## Building and Running

To validate our ideas we offer this implementation where PCNN acts as the fact extractor, and MINERVA as the reasoner. You can implement your trials with other models in a similar fashion.

### Prerequisites

To properly run this code, following packages of Python 3 with the newest available version should be prepared:

```text
    tqdm (for progress display)
    tensorflow
    numpy,scipy,scikit-learn
```

### Running

You can use the following command to run this code:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 joint_trainer.py \
            --total_iterations=500 \
            --use_replay_memory=1 \
            --train_pcnn=1 \
            --bfs_iteration=200 \
            --use_joint_model=1 \
            --pcnn_dataset_base="data" \
            --pcnn_dataset_name="FB60K+NYT10/text" \
            --pcnn_max_epoch=250 \
            --base_output_dir="output" \
            --gfaw_dataset_base="data" \
            --gfaw_dataset="FB60K+NYT10/kg" \
            --load_model=1 \
            --load_pcnn_model=1 \
            --batch_size=64 \
            --hidden_size 100 --embedding_size 100 \
            --random_seed=55 \
            --eval_every=100 \
            --model_load_dir="experiments/FB60K+NYT10/kg/3_100_100_0.01_83_True_200_400_02130056/model"

```
This offers you the parameters that can yield the best results while not consuming too much time; it designates two GPUs to separately store the two agents.

Firstly, the MINERVA model is trained for the number of iterations designated by **--total_iterations**; then the pcnn model is trained by a default of 200 iterations; 
finally the two agents are trained together for another **--total_iterations** iterations, in which the first **--bfs_iteration** iterations is trained using BFS aid.

### Parameters

    total_iteations: total iterations for the model to run

    use_replay_memory: usage of replay memory or not

    train_pcnn: train pcnn in the process of joint reasoning or use a static model

    bfs_iteration: first how many iterations are for BFS path search

    use_joint_model: usage of joint model or not

    pcnn_dataset_base: base folder containing all relation extraction corpuses

    pcnn_dataset_name: which corpus dataset to use

    base_output_dir: what directory to save output to

    gfaw_dataset_base: base folder containing all KG data

    gfaw_dataset: which KG dataset to use 

    load_model: whether to load a previously trained model (MINERVA) first; if there isn't any choose false/0

    load_pcnn_model: whether to load a previously trained PCNN model first

    batch_size, hidden_size, embedding_size: evidently

    random_seed: fix a random seed. RL methods heavily fluctuate in sircumstances

    eval_every: enter evaluation session (use validation set) for every what epochs

    model_load_dir: directory where the model will be loaded


You can find results in the experiments folder's scores.txt, under each model labeling its parameters.

Results will be saved to a folder named "experiments" as a subdirectory. Another folder "_preprocessed_data" will save the preprocessed datasets for quicker code running. 

## Performance

### Datasets

We offer two datasets: FB15K-NYT10 and UMLS-PubMed. Download it here, with some data dealing toolkits: 

https://drive.google.com/file/d/1hCyPBjywpMuShRJCPKRjc7n2vHpxfetg/view?usp=sharing

if you want to create your own dataset, it should look like this:

the knowledge graph dataset should look like ones for [MINERVA](https://github.com/shehzaadzd/MINERVA). take a look: [Here](https://github.com/shehzaadzd/MINERVA/tree/master/datasets/data_preprocessed/FB15K-237)

the corpus should look like ones for [OpenNRE](https://github.com/thunlp/OpenNRE). An example is below:

```
[
    {
        'sentence': 'Bill Gates is the founder of Microsoft .',
        'head': {'word': 'Bill Gates', 'id': 'm.03_3d', ...(other information)},
        'tail': {'word': 'Microsoft', 'id': 'm.07dfk', ...(other information)},
        'relation': 'founder'
    },
    ...
]
```

### Baselines

baselines are offered in a subfolder. Due to the size of the datasets, you may have to preprocess the datasets in the manner these baselines require them. 

Toolkits, as mentioned before, are offered in the dataset zip file.

MINERVA : included as a part of the model.

Two-Step : Use PCNN to extract relations on a given corpus dataset beforehand, add them to the knowledge graph's training set by a confidence threshold, and train MINERVA.

[ConvE](https://github.com/INK-USC/CPL/blob/master/baselines/ConvE-master/) : SOTA KG embedding-based reasoning method.

[OpenKE](https://github.com/INK-USC/CPL/tree/master/baselines/OpenKE-master) : Classical embedding baselines prior to 2014. Implemented by THU NLP group.
Includes TransE, DistMult and ComplEx.

[JointNRE](https://github.com/INK-USC/CPL/tree/master/baselines/JointNRE-master) : A model that conducts relation extraction with the aid of KGs. 
We consulted the author and acquired a release that saves the trained embedding and uses OpenKE-TransE to conduct the testing task.

[MultiHop](https://github.com/INK-USC/CPL/tree/master/baselines/MultiHopKG-master) : SOTA path-based reasoning method. This model uses PyTorch.

[TransE + LINE](https://github.com/INK-USC/CPL/tree/master/baselines/triple%2Btext) : An implementation of the same idea by one of our authors. 
It uses LINE for relation extraction, and TransE for KG embedding.


### Results

![Results1](https://github.com/INK-USC/CPL/blob/master/U%2BP%20results.png)

![Results2](https://github.com/INK-USC/CPL/blob/master/F%2BN%20results.png)

We conducted tests on different levels of fractions of the original knowledge graphs. The results corresponding to the scope of part of knowledge graphs for different models are as follows:

![Fact-Select](https://github.com/INK-USC/CPL/blob/master/fact-select.png)

## File Structure

/nrekit: the Relation Extraction Agent kit.

/rl_code: the reasoner.

/model/trainer.py: training code.

/model/tester.py: testing code.

/joint_trainer.py: main running code. loads both models and runs sessions.

/pure_gfaw.py: a code without pcnn interference, as a baseline against our model.

## TODO

- [ ] Unify baseline input and output to current MINERVA style
- [ ] Improve stability of BFS process, make it work on any dataset
- [ ] Improve code style, add comments to code files and data dealers
- [ ] Migrate to PyTorch


