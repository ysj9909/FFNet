python -u run.py --is_training 1 --model_id ETTh1 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --ffn_ratio 2 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.7 --itr 1 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTh1 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 192 --ffn_ratio 2 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.7 --itr 1 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTh1 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 336 --ffn_ratio 2 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.7 --itr 1 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id ETTh1 --model FFNet --root_path data/long_term_forecast/ETT-small --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 720 --ffn_ratio 2 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 7 --dropout 0.7 --itr 1 --train_epochs 100 --batch_size 512 --patience 20 --learning_rate 0.0001 --des Exp --lradj type3 --use_multi_scale False --small_kernel_merged False