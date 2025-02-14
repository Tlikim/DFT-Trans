# DFT-Trans
This is the official implementation of paper "DFT-Trans: A Bidirectional Encoder for Efficient Fusion of Time-Frequency Domain Textual Features"  
## Pretrain Corpus
We are providing the corpus used to pretrain the model.  
[C4](https://huggingface.co/datasets/allenai/c4), [Wikipeidia](https://huggingface.co/datasets/wikipedia) and 
[BookCorpus](https://huggingface.co/datasets/bookcorpus): These datasets are used for Bert, Albert. We can obtained from [huggingface](https://huggingface.co/).  
C4BookC: This dataset is used to pretrain the DFT-Trans model. Approximately 40 million sentences. Consists of part of the C4 dataset and the BookCorpus dataset used to train our model. As the pretraining corpus is so large, we will publish it later.
## Model Pretrain
The pretraining code is stored in the model_pretrain folder. You can store the pre-training corpus in the data file and run the command:  
``python3 model_pretrain.py``  
We will further open source our model parameter weights in the following.
## Glue Train
The pretrained model will be tested for performance on the GLUE benchmark and the SQuAD dataset.  
The code to perform downstream task fine-tuning is stored separately in the Glue_train folder. We can fine-tune the model on the GLUE benchmark by the following command ``sh train_glue.sh``. Similarly we can fine-tune the model on the SQuAD dataset by the following command 
``sh train_squad1.sh``.  
Both the GLUE benchmark and the SQuAD dataset need to be stored in the data folder.


