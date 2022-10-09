## Model
> ### BERT4Rec
- [paper](https://arxiv.org/abs/1904.06690v2)
- [Model reference code](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch)

## How to run

1. Using your own dataset.
    ```bash
    ├── data
      ├── train
      ├── test
    ```

2. Main Training
   ```
   python run_train.py --model_name BERT4Rec 
   ```

3. Inference
   ```
   python inference.py --model_name BERT4Rec 
   ```
