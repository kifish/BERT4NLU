# BERT4NLU
Bert-For-NLU-Tasks


### text-classification


#### bert for chatbot

training parameters: src/config/config.py

After finetuning BERT for 6000 steps, evaluation results on the PersonaChat valid_self_original dataset are rather good:
```
acc : 0.9756
R1: 0.8051
R2: 0.9025
R5: 0.9670
```

After finetuning BERT for 3 epochs, evaluation results on the PersonaChat valid_self_original dataset are rather awesome:
```
acc : 0.9791
R1: 0.8548
R2: 0.9313
R5: 0.9812
```

Moreover, evaluation results on the PersonaChat test_self_original dataset achieve SOTA:
```
acc : 0.9783
R1: 0.8501
R2: 0.9370
R5: 0.9855
```


Baseline:      
[Personalizing Dialogue Agents: I have a dog, do you have pets too?
](https://arxiv.org/abs/1801.07243)





### sequence-labeling

https://github.com/kifish/NER-demo/tree/bert


### sequence-ranking
To be added.



