export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_src/do_generate.py \
       --model_ckpt meta-llama/Llama-3.2-3B-Instruct \
       --dataset_name GSM8K \
       --test_json_filename test_all \
       --note tensor_parallelism \
       --num_subquestions 5 \
       --num_a1_steps 5 \
       --disable_a5 \
       --start_idx 0 \
       --end_idx 4 \
       --num_rollouts 16 \
       --api vllm \
       --model_parallel \
       --tensor_parallel_size 4
