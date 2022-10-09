## Model
> ### BERT4Rec
- [paper](https://arxiv.org/abs/1904.06690v2)
- [Model reference code](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)

## How to run

1. Preparing your datasets
    ```bash
    ├── data
      ├── train
      ├── <span style="color:black">test</span>     
    ```

2. Main Training
   ```
   python run_train.py --model_name BERT4Rec 
   ```

3. Inference
   ```
   python inference.py --model_name BERT4Rec 
   ```
