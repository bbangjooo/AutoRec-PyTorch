### Files

- `data.py` : custom dataset 클래스
- `models.py` : AutoRec 모델과 MaskedRMSE 클래스

### Data

[ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) 출처 : http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

### example

```
> python .\run.py --help
usage: run.py [-h] [--epochs EPOCHS] [--batch BATCH] [--lr LR] [--wd WD] [--ksize KSIZE]

AutoRec with PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -e EPOCHS
  --batch BATCH, -b BATCH
  --lr LR, -l LR        learning rate
  --wd WD, -w WD        weight decay(lambda)
  --ksize KSIZE, -k KSIZE
                        hidden layer feature_size
                         
> python .\run.py --epochs 20 --batch 32 --wd 0.001 -k 200
```

### Result 

![autoRec](https://user-images.githubusercontent.com/51329156/106747396-357d4980-6667-11eb-9b75-796767ce92de.png)

논문에서 제시하는 파라미터대로 했는데 loss가 3 이하로 내려가지 않는다. 이유는 아직 잘 모르겠다..