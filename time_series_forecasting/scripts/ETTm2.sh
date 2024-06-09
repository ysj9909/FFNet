python -u run.py --is_training 1 --model_id ETTm2 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 96 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.2 --enc_in 7 --dropout 0.6 --itr 1 --train_epochs 100 --batch_size 2048 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTm2 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 192 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.2 --enc_in 7 --dropout 0.6 --itr 1 --train_epochs 100 --batch_size 2048 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTm2 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 336 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.2 --enc_in 7 --dropout 0.7 --itr 1 --train_epochs 100 --batch_size 2048 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTm2 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTm2.csv --data ETTm2 --features M --seq_len 96 --pred_len 720 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.1 --enc_in 7 --dropout 0.6 --itr 1 --train_epochs 100 --batch_size 2048 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False


