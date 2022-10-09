

import argparse
import os
import wandb
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import BERT4RecDataset
from models import BERT4RecModel
from trainers import BERT4RecTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="BERT4Rec", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--using_pretrain", action="store_true")
    parser.add_argument("--mask_p", type=float, default=0.15, help="mask probability")
    parser.add_argument("--rm_position", action="store_true", help="remove position embedding")
    parser.add_argument("--wandb_name", type=str)

    # 1. wandb init
    #wandb.init(project="movierec_train_styoo", entity="styoo", name="SASRec_WithPretrain")
    args = parser.parse_args()
    wandb.init(project="menu_train")
    wandb.run.name = args.wandb_name

    # 2. wandb config
    wandb.config.update(args)
    print(str(args))

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "menu_final.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file, args.model_name
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    if args.model_name == 'BERT4Rec':
        train_dataset = BERT4RecDataset(args, user_seq, data_type="train")
        eval_dataset = BERT4RecDataset(args, user_seq, data_type="valid")
        test_dataset = BERT4RecDataset(args, user_seq, data_type="test")

    elif args.model_name != 'BERT4Rec':
        print("---------------You cannot use it!-------------------")



    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )
    if args.model_name == 'BERT4Rec':
        model = BERT4RecModel(args=args)

        trainer = BERT4RecTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, None, args
        )
    
    elif args.model_name == 'BERT4Rec':
        model = BERT4RecModel(args=args)

        trainer = BERT4RecTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, None, args
        )
    print(f"Not using pretrained model. The Model is same as {args.model_name}")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    for epoch in tqdm(range(args.epochs)):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)
        
        # 3. wandb log
        wandb.log({"recall@1" : scores[0], 
                   "ndcg@1" : scores[1],
                #    "recall@5" : scores[0], 
                #    "ndcg@5" : scores[1], 
                #    "recall@10" : scores[2],
                #    "ndcg@10" : scores[3]}
                   "recall@4" : scores[2],
                   "ndcg@4" : scores[3]})
                   
        # early_stopping(np.array([scores[2]]), trainer.model)
        early_stopping(np.array([scores[0]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


if __name__ == "__main__":
    main()
