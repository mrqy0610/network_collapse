
python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--min_nodes 30 --max_nodes 50 \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-2 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --eval_graph_path './data/Crime.graphml' \
--add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/hierarchy_a2c/a2c_sameWeights_noLimit_epsInit0.1NoDecay_normalReward'\
'_useHuberLoss_useReconsLoss_useCrime_usePairWise_gamma0.95_actorLr1e-2/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--min_nodes 30 --max_nodes 50 \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-3 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --eval_graph_path './data/Crime.graphml' \
--add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/hierarchy_a2c/a2c_sameWeights_noLimit_epsInit0.1NoDecay_normalReward'\
'_useHuberLoss_useReconsLoss_useCrime_usePairWise_gamma0.95_actorLr1e-3/'


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--min_nodes 30 --max_nodes 50 \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-4 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --eval_graph_path './data/Crime.graphml' \
--add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/hierarchy_a2c/a2c_sameWeights_noLimit_epsInit0.1NoDecay_normalReward'\
'_useHuberLoss_useReconsLoss_useCrime_usePairWise_gamma0.95_actorLr1e-4/'


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--min_nodes 30 --max_nodes 50 \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-5 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --eval_graph_path './data/Crime.graphml' \
--add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/hierarchy_a2c/a2c_sameWeights_noLimit_epsInit0.1NoDecay_normalReward'\
'_useHuberLoss_useReconsLoss_useCrime_usePairWise_gamma0.95_actorLr1e-5/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--min_nodes 30 --max_nodes 50 \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --eval_graph_path './data/Crime.graphml' \
--add_eval_stage --eval_freq_in_train 10 \
--save_model --output_res \
--output_res_dir './res/hierarchy_a2c/a2c_sameWeights_noLimit_epsInit0.1NoDecay_normalReward'\
'_useHuberLoss_useReconsLoss_useCrime_usePairWise_gamma0.95_actorLr1e-6/'


#################################################################################### new

############## node 50

python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_1/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_1/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_2/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_2/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_double_node20_1/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_doubleGraphNode20_1/'


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_3/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_3/'





python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_4/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_4/'


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_5/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_5/'


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_6/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_6/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node20_7/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode20_7/'


############## node 20


python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node50_1/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode50_1/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_single_node50_2/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_singleGraphNode50_2/'



python main.py --algorithm_name 'hie-policy' --connectivity_method 'pairwise' --lower_agent_type 'degree_greedy' \
--reward_normalization --critic_use_huber_loss --huber_loss_delta 10.0 --eliminate_orphan_node \
--graphsage_inner_dim 128 --graphsage_output_dim 128 --graphsage_adj_num_samples 10 --qnetwork_inner_dim 128 \
--attention_embedding_dim 128 --n_encode_layers 2 --max_upper_actions_len 114514 \
--random_stop_actions --random_stop_eps 0.1 \
--total_episodes 10000 --memory_capacity 2000 --model_update_freq 100 \
--batch_size 128 --reward_gamma 0.95 --actor_learning_rate 1e-6 --critic_learning_rate 5e-4 \
--add_graph_reconstruction_loss --graph_reconstruction_loss_alpha 0.001 \
--eval_episodes 1 --add_eval_stage --eval_freq_in_train 10 --save_model --output_res \
--train_with_preload_graph --train_graph_path './data/data_set_double_node50_1/' \
--output_res_dir './res/hierarchy_a2c/a2c_useHuberLoss_useReconsLoss_usePairWise_doubleGraphNode50_1/'



pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gym==0.25.1




