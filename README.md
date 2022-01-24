# NodeInGraph

We first generate node embedding as LINE [1] with the open source code: https://github.com/tangjianpku/LINE

Then we calculate the similarity based on node embedding of the head and tail node as the prediction score. (as shown in model_predict.py)

Instruction to produce the outputs:

## 1. generate the input file of LINE

source_id1 dest_id1 weight1

source_id2 dest_id2 weight2

source_id3 dest_id3 weight3

...

Note that, the weight is the occurrence frequency of the source node and the destination node.

## 2. generate LINE embedding 

Clone the open source code of LINE from https://github.com/tangjianpku/LINE.

And run LINE embedding generation process in shell:

./line -train train_file -output embedding_file -binary 1 -size 100 -order 1 -negative 5 -samples 100 -rho 0.025 -threads 10

## 3. prediction the result

Run code in model_predict.py to calculate the dot similarity of two target nodes.

## 4. reference
[1] LINE: Large-scale Information Network Embedding.
