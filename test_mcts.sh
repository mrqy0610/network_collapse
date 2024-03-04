python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_multi_node12/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode12_1/'



python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_1/'



python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_2/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_2/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_double_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_doubleGraphNode20_1/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_3/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_3/'



python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_4/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_4/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_5/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_5/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_6/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_6/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_7/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res \
--output_res_dir './res/mcts/mcts_usePairWise_gamma0.95_actorLr2e-3_singleGraphNode20_7/'



######################################### test new


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_node_value_normal \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_valueNorm/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_dynamic_c_puct \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_dynamicCpuct/'


python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_ucbAddReward/'

########### combine

python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_dynamic_c_puct \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_ucbAddReward_dynamicCpuct/'

python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_ucbAddReward_valueNorm/'

python main.py --algorithm_name 'mcts' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_episodes 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--output_res_dir './res/mcts/test_trick/mcts_singleGraphNode20_1_ucbAddReward_valueNorm_dynamicCpuct/'

########### multi

python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_1/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode20_1/'


python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node50_1/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 2000 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode50_1_capacity2k/'


python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node20_2/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 200 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode20_2/'


python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node50_1/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 2000 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--load_model --load_model_path "/home/LAB/qiuyue/network_collapse/res/mcts_multi/"\
"mcts_multi_combineTrick_singleGraphNode50_1_capacity2k_notTrainDelay/model/model_2839.pth" \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode50_1_capacity2k_notTrainDelay_1/'







python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node50_1/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 2000 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode50_1_capacity2k_notTrainDelay_resetTree_newNoiseMethod/'


python main.py --algorithm_name 'mcts-multi' --connectivity_method 'pairwise' \
--train_with_preload_graph --train_graph_path "/home/LAB/qiuyue/network_collapse/data/data_set_single_node50_2/" \
--reward_reflect --learning_rate 2e-3 --total_training_steps 20000 --memory_capacity 2000 \
--batch_size 128 --reward_gamma 0.95 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model_freq 10 \
--save_model --output_res --mcts_ucb_add_reward --mcts_node_value_normal --mcts_dynamic_c_puct \
--mcts_num_actors 32 --mcts_para_update_freq 10 \
--output_res_dir './res/mcts_multi/mcts_multi_combineTrick_singleGraphNode50_2_capacity2k_notTrainDelay_fix/'






