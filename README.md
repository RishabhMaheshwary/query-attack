## A Strong Baseline for Query Efficient Attacks in a Black Box Setting

This repository contains source code for the research work described in our EMNLP 2021 paper:

[A Strong Baseline for Query Efficient Attacks in a Black Box Setting](https://arxiv.org/abs/2109.04775)

The attack jointly leverages attention mechanism and Locality-sensitive hashing (LSH) for word ranking. It has been [implemented](https://github.com/RishabhMaheshwary/TextAttack/blob/a65831810cffb62ce25ad0acf556315e08a07d85/textattack/search_methods/greedy_word_swap_wir.py#L60) in the [Textattack](https://github.com/RishabhMaheshwary/TextAttack/tree/query-attack) framework so as to ensure consistent comparison with other attack methods.


### Installation

1. Clone the repository using the ```recursive``` flag so as to set up the Textattack submodule.

   ```git clone --recursive https://github.com/RishabhMaheshwary/query-attack.git```
   
2. Make sure ```git lfs``` is installed in your system. If not installed refer [this](https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear).

3. Run the below commands to download pre-trained attention models.
   
      ```git install lfs```
      
      ```git lfs pull ```

4. It is recommended to create a new conda environment to install all dependencies.

    ```
        cd Textattack
        pip install -e .
        pip install allennlp==2.1.0 allennlp-models==2.1.0 
        pip install tensorflow 
        pip install numpy==1.18.5
    ```

### Running query-attack

1. To attack BERT model trained on IMDB using the WordNet search space use the following command:

    ```
    textattack attack \
    --recipe lsh-with-attention-wordnet \
    --model bert-base-uncased-imdb \
    --num-examples 500 \
    --log-to-csv outputs/ \
    --attention-model attention_models/yelp/han_model_yelp
    ```
Note: The attention model specified should be trained on a different dataset than that of the target model. This is because in the black box setting we do not have access to the training data of the target model.

2. To attack LSTM model trained on Yelp using the WordNet search space use the following command:
    
    ```
    textattack attack \
    --recipe lsh-with-attention-wordnet \
    --model lstm-yelp \
    --num-examples 500 \
    --log-to-csv outputs/ \
    --attention-model attention_models/imdb/han_model_imdb
    ```

3. To evaluate BERT model trained on MNLI using the HowNet search space use the following command:

    ```
    textattack attack \
    --recipe lsh-with-attention-hownet \
    --model bert-base-uncased-mnli \
    --num-examples 500 \
    --log-to-csv outputs/ \
    --attention-model mnli
    ```
The tables below shows what arguments to pass to ```--model``` flag and ```--recipe``` flag in the textattack command to attack BERT and LSTM models on IMDB, Yelp and MNLI datasets across various search spaces.
<table>
<tr><td>

|   Model   |         --model flag         |    
|:---------:|:----------------------------:|
| BERT-imdb | ```bert-base-uncased-imdb``` |
| BERT-yelp | ```bert-base-uncased-yelp``` |
| BERT-mnli | ```bert-base-uncased-mnli``` |
| LSTM-imdb |        ```lstm-imdb```       |
| LSTM-yelp |        ```lstm-yelp```       |
| LSTM-mnli |        ```lstm-mnli```       |

</td>
<td>

| Search Space |              --recipe flag             |
|:------------:|:--------------------------------------:|
|    WordNet   |    ```lsh-with-attention-wordnet```    |
|    HowNet    |     ```lsh-with-attention-hownet```    |
|   Embedding  |   ```lsh-with-attention-embedding```   |
| Embedding+LM | ```lsh-with-attention-embedding-gen``` |

</td></tr>

</table>

To run the baselines in the paper refer to the [main Textattack repository](https://github.com/QData/TextAttack).    
    

### Training attention models


1. ```pip install gensim==3.8.3 torch==1.7.1+cu101```

2. The datasets used to train the attention model can be found [here](https://drive.google.com/drive/folders/1c-fW4Gq849j587jC2lK791JIkEVArsFn?usp=sharing).

3. Unzip the dataets and specify the path of the dataset in the ```create_input_files.py``` file.

4. The model then can be trained using the command below:
```
python create_input_files.py
python train.py
python eval.py
```
The implementation of the training attention models is borrowed from [here](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification).

5. For NLI task the attention weights are computed using the pre-trained decomposable attention model from [AllenNLP api](https://demo.allennlp.org/textual-entailment/elmo-snli).
