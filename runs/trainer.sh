export CUDA_VISIBLE_DEVICES=5

python symba_trainer.py --experiment_name="BART_check" --optimizer_lr=0.0001 --optimizer_weight_decay=0.00009 \
                        --batch_size=64 --epoch=10 --dataset_name="QED_Amplitude" --clip_grad_norm=1 --scheduler_type="multi_step" \
                        --scheduler_milestones 20 40 60 --model_name="bart-base" --maximum_sequence_length=256
