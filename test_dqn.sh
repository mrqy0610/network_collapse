python main.py --algorithm_name 'dqn' --connectivity_method 'gcc' \
--fix_graph --fix_graph_type --graph_type 'barabasi_albert' \
--min_nodes 10 --max_nodes 30 --reward_reflect \
--graphsage_inner_dim 128 --graphsage_output_dim 64 --graphsage_adj_num_samples 10 \
--action_embed_dim 64 --qnetwork_inner_dim 128 \
--learning_rate 5e-3 --total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --epsilon_greedy_rate 0.9 --target_q_update_freq 100 --reward_gamma 0.9 \
--save_model --output_res --output_res_dir './res/baseline_dqn/dqn_test_sameGraph_sameWeights_reflectReward/'