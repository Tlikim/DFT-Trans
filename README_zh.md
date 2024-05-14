# DFT-Trans
[English](./README.md)  
这是论文"DFT-Trans: Pretraining Cost Reduction via Domain Mixing Tokens of Time and Frequency for NLP Tasks"的官方代码实现。  
## Pretrain Corpus
[C4](https://huggingface.co/datasets/allenai/c4), [Wikipeidia](https://huggingface.co/datasets/wikipedia) and 
[BookCorpus](https://huggingface.co/datasets/bookcorpus): 这些训练语料都是用于Bert，Albert模型的预训练，可以从[huggingface](https://huggingface.co/)中获取。  
[C4BookC](https://pan.baidu.com/s/1BxkE75I9iKkWppkTFBWj2w?pwd=5ls4): 这个数据集用于预训练DFT-Trans模型。大约有40000000条句子。这是我们从C4和BookCorpus数据集中提取出来的。
