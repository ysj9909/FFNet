python -u run.py --is_training 1 --model_id Traffic --model FFNet --root_path data/long_term_forecast/traffic/ --data_path traffic.csv --data custom --features M --seq_len 96 --pred_len 96 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 862 --dropout 0.9 --itr 1 --train_epochs 100 --batch_size 128 --patience 5 --learning_rate 0.00005 --des Exp --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id Traffic --model FFNet --root_path data/long_term_forecast/traffic/ --data_path traffic.csv --data custom --features M --seq_len 96 --pred_len 192 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 862 --dropout 0.9 --itr 1 --train_epochs 100 --batch_size 128 --patience 5 --learning_rate 0.00005 --des Exp --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id Traffic --model FFNet --root_path data/long_term_forecast/traffic/ --data_path traffic.csv --data custom --features M --seq_len 96 --pred_len 336 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 862 --dropout 0.9 --itr 1 --train_epochs 100 --batch_size 128 --patience 5 --learning_rate 0.00005 --des Exp --use_multi_scale False --small_kernel_merged False

python -u run.py --is_training 1 --model_id Traffic --model FFNet --root_path data/long_term_forecast/traffic/ --data_path traffic.csv --data custom --features M --seq_len 96 --pred_len 720 --ffn_ratio 12 --patch_size 4 --patch_stride 2 --num_blocks 1 --large_size 51 --small_size 5 --dims 64 64 64 64 --head_dropout 0.0 --enc_in 862 --dropout 0.9 --itr 1 --train_epochs 100 --batch_size 128 --patience 5 --learning_rate 0.00005 --des Exp --use_multi_scale False --small_kernel_merged False