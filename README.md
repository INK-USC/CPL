# CPL: Collaborative Policy Learning

This is the code for paper "Collaborative Policy Learning for Open Knowledge Graph Reasoning".

## Our Idea

- Task
  - Given query (e_1, r)
  - predict a certain length $t$ path on the graph (e_1, r_{12}, e_2, r_{23}, ..., e_{t}, r_{t(t+1)},e_{t+1})
  - make sure that (e_1, r, e_{t+1}) is a fact in the knowledge base

- Graph: Use Go For A Walk to predict the path

- Text: Use text information to find ***inexistent*** edges, further help predict paths on the graph

  - ***Methods:*** Use certain sentence embedding method to model the text sentence into a embedding, after that we use an agent called ***Sentence Relation Selector*** to help us select a most relevant sentence at time step t, then split the sentence embedding into three parts: ![](http://latex.codecogs.com/gif.latex?{v_{e_t},v_{e_{t+1}}, r_{Sentence}})

  - Example and motivation:

    - ![image-20181020120019098](https://github.com/shanzhenren/GraphPath/blob/master/README.assets/image-20181020120019098.png) 

    - When given a query *(Melinda, live in, ?)*, if we only got the graph information, we may find the path *(Melinda --- friend --- Jane Doe --- born --- Seattle)*, which is not correct. 

    - With the text information, we may choose a path like *(Melinda --- wife --- Bill Gates --- chair --- Microsoft --- headquarter in --- Seattle)*. In time step 1, the fact *(Melinda, wife, Bill Gates)* comes from the sentence 

      >  *William H. Gates, and his wife Melinda gave $3.3B to their two foundation*.

- Graph+Texts

![image-20181020190951048](https://github.com/shanzhenren/GraphPath/blob/master/README.assets/image-20181020190951048.png)

- Take-away message:
  - Use text information to add potential useful ***inexistent*** edges
  - Also use RL to give rewards to ***Sentence Relation Selector***.

## Running

You can use the following command to run this code:

```
CUDA_VISIBLE_DEVICES=0,1 python3 joint_trainer.py \
            --total_iterations=400 \
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

usable configurations:

    total_iteations: total iterations for the model to run

    use_replay_memory: usage of replay memory or not

    train_pcnn: train pcnn in the process of joint reasoning or use a static model

    bfs_iteration: first how many iterations are for BFS path search

    use_joint_model: usage of joint model or not

    pcnn_dataset_base: base folder containing all relation extraction corpuses

    pcnn_dataset_name: which to use

    base_output_dir: what directory to save output to

    gfaw_dataset_base: base folder containing all KG data

    gfaw_dataset: which to use 

    load_model: whether to load a previously trained model (MINERVA) first

    load_pcnn_model: whether to load a previously trained PCNN model first

    batch_size, hidden_size, embedding_size: evidently

    random_seed: fix a random seed. RL methods heavily fluctuate in sircumstances

    eval_every: enter evaluation session (use validation set) for every what epochs

    model_load_dir: directory where the model will be loaded


You can find results in the experiments folder's scores.txt, under each model labeling its parameters.

Results will be saved to a folder named "experiments" as a subdirectory. Another folder "_preprocessed_data" will save the preprocessed datasets for quicker code running. 

## Datasets

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

## File Structure

/nrekit: the Relation Extraction Agent kit.

/rl_code: the reasoner.

/model/trainer.py: training code.

/model/tester.py: testing code.

/joint_trainer.py: main running code. loads both models and runs sessions.

/pure_gfaw.py: a code without pcnn interference, as a baseline against our model.

## Cite this paper


