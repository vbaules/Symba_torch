import torch
import argparse
from engine import Trainer, Config

parser = argparse.ArgumentParser(description="SYMBA Training")
parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiemnt")
parser.add_argument('--model_name', type=str, default="seq2seq_transformer", help="Name of the model eg: seq2seq_transformer, BART, mBART")
parser.add_argument('--dataset_name', type= str, default="QCD", help = "Name of the dataset: eg: QCD, QED")
parser.add_argument('--epochs', type=int, default=90, help="Number of epochs to train")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--optimizer_type', type=str, default="adam", help="Optimizer to be used")
parser.add_argument('--optimizer_lr', type=float, default=0.0001, help="Learning rate of optimizer")
parser.add_argument('--optimizer_weight_decay', type=float, default=0.0001, help="Weight decay")
parser.add_argument('--clip_grad_norm', type=float, default=-1, help="Set the clipping value for gradient")
parser.add_argument('--scheduler_type', type=str, default="none", help="Name of the scheduler")
parser.add_argument('--scheduler_milestones', nargs="+", type=int, default=[40, 60], help="Milestones for the scheduler")
parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="Gamma for the scheduler")
parser.add_argument('--vocab_size', type=int, default=2875, help="Set the size of vocabulary")
parser.add_argument('--maximum_sequence_length', type=int, default=256, help="Set the length of sequence")
parser.add_argument('--embedding_size', type=int, default=512, help="Set the size of embedding")
parser.add_argument('--hidden_dim', type=int, default=16384, help="Set the hidden dimension")
parser.add_argument('--num_encoder_layers', type=int, default=3, help="Number of encoder layers")
parser.add_argument('--num_decoder_layers', type=int, default=3, help="Number of decoder layers")
parser.add_argument('--num_head', type=int, default=8, help="Number of heads in transformer")
parser.add_argument('--dropout', type=float, default=0.1, help="Set the Dropout in the network")
parser.add_argument('--label_smoothing', type=float, default=0, help="Set the value of label smoothing in loss function")

args = parser.parse_args()

config = Config(
    experiment_name=args.experiment_name,
    root_dir="./",
    device="cuda",
    model_name=args.model_name,
    dataset_name=args.dataset_name,
    epochs=args.epochs,
    seed=args.seed,
    training_batch_size=args.batch_size,
    test_batch_size=2*args.batch_size,
    optimizer_type=args.optimizer_type,
    optimizer_lr=args.optimizer_lr,
    optimizer_weight_decay=args.optimizer_weight_decay,
    clip_grad_norm=args.clip_grad_norm,
    scheduler_type=args.scheduler_type,
    scheduler_milestones=args.scheduler_milestones,
    scheduler_gamma=args.scheduler_gamma,
    vocab_size=args.vocab_size,
    maximum_sequence_length=args.maximum_sequence_length,
    embedding_size=args.embedding_size,
    hidden_dim=args.hidden_dim,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    nhead=args.num_head,
    dropout=args.dropout
)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

class SymbaTrainer(Trainer):
    def criterion(self, y_pred, y_true):
        return loss_fn(y_pred, y_true)
    
if __name__ == "__main__":
    trainer = SymbaTrainer(config)
    trainer.fit()
    
