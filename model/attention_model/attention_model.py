import math
import torch
import numpy as np
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint

from typing import NamedTuple
# from utils.functions import sample_many
# from utils.beam_search import CachedLookup
# from utils.tensor_functions import compute_in_batches
from model.attention_model.attention_utils import MultiStepsRecoder
from model.attention_model.graph_encoder import GraphAttentionEncoder


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    node_embeddings: Node embeddings for fixed
        torch.Size([batch_size, graph_size + 1, embedding_dim])
    context_node_projected: The fixed context projection of the graph embedding
        torch.Size([batch_size, 1, embedding_dim])
    glimpse_key: Key computed from node embedding
        torch.Size([num_heads, batch_size, 1, graph_size + 1, sub_embedding_dim])
    glimpse_val: Val computed from node embedding
        torch.Size([num_heads, batch_size, 1, graph_size + 1, sub_embedding_dim])
    logit_key: Key to calculate choosing logits(with query as output of decoder)
        torch.Size([batch_size, 1, graph_size + 1, embedding_dim])
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(
            self, config, embedding_dim, n_encode_layers=2,
            tanh_clipping=10., mask_inner=True, mask_logits=True,
            normalization='batch', n_heads=8):
        super(AttentionModel, self).__init__()
        # set basic config for attention model
        self.config = config
        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None  # action generate type: sample or argmax
        self.temp = 1.0

        # config may not be used
        # self.hidden_dim = hidden_dim
        # self.allow_partial = problem.NAME == 'sdvrp'
        # self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        # self.is_orienteering = problem.NAME == 'op'
        # self.is_pctsp = problem.NAME == 'pctsp'
        # self.problem = problem
        # self.shrink_size = shrink_size  # shrink finish batch

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner  # mask prob when compute compatibility
        self.mask_logits = mask_logits  # mask prob when compute logits
        self.n_heads = n_heads

        self.step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        self.node_dim = self.config.feature_num  # Embedding dim for each node

        # Learned input symbols for first and last action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.encoder = GraphAttentionEncoder(
            config=config,
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(self.step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )

        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        first_a = state.get_first_node()
        batch_size, num_steps = current_node.size()
        """
        current_node torch.Size([batch_size, 1])
        first_a torch.Size([batch_size, 1])
        """
        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.get_steps() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _get_attention_node_data(self, fixed, state):
        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        compatibility:
            torch.Size([n_heads, batch_size, num_steps, 1, graph_size + 1])
        glimpse_Q:
            torch.Size([n_heads, batch_size, num_steps, 1, sub_embed_dim])
        glimpse_K:
            torch.Size([n_heads, batch_size, num_steps, graph_size + 1, sub_embed_dim])
        glimpse_V:
            torch.Size([n_heads, batch_size, num_steps, graph_size + 1, sub_embed_dim])
        logit_K:
            torch.Size([batch_size, num_steps, graph_size + 1, embed_dim])
        mask:
            torch.Size([batch_size, num_steps, graph_size + 1])
        """

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        """
        heads torch.Size([n_heads, batch_size, num_steps, 1, sub_embed_dim])
        glimpse torch.Size([batch_size, num_steps, 1, embed_dim])
        final_Q torch.Size([batch_size, num_steps, 1, embed_dim])
        logit_K torch.Size([batch_size, num_steps, graph_size, embed_dim])
        logits: torch.Size([batch_size, num_steps, graph_size + 1])
        """
        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = project(graph_embedding) + project(first_node, last_node)
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            # randomly choose to end actions for eps
            if self.config.random_stop_actions and np.random.rand(1)[0] < self.config.random_stop_eps and probs[:, -1] > 0:
                return torch.tensor([probs.shape[-1] - 1]).to(self.config.device)
            selected = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        return selected

    def _calc_log_likelihood(self, _log_p, a, mask):
        """
        :param _log_p: torch.Size([batch_size, seq_len, node_num])
        :param a: torch.Size([batch_size, seq_len])
        :param mask:
        :return:
        """
        # choose_vector, index = torch.zeros_like(_log_p), a.unsqueeze(-1)
        # choose_vector.scatter_(2, index, torch.ones_like(index).to(dtype=torch.float32))
        # log_p_new = (_log_p * choose_vector).sum(-1)
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def decoder(self, features, nodes, embeddings):
        """
        features torch.Size([batch_size, graph_size, node_dim])
        nodes torch.Size([batch_size, graph_size])
        embeddings torch.Size([batch_size, graph_size, embed_dim])
        """
        outputs = []
        sequences = []
        # state = self.problem.make_state(input)
        state = MultiStepsRecoder(config=self.config, nodes=nodes)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)
        # Perform decoding steps
        while True:
            """
            log_p torch.Size([batch_size, step_size, graph_size + 1])
            mask torch.Size([batch_size, step_size, graph_size + 1])
            """
            # calculate logprob for selecting nodes
            log_p, mask = self._get_log_p(fixed, state)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            # print('log_p', log_p[:, 0, :])
            # print('selected', selected)

            # update state after selecting a node
            state.update(selected)

            # collect action and prob of step(if not reach end of node)
            if not state.reach_end_of_token():
                outputs.append(log_p[:, 0, :])
                sequences.append(selected)
            # only collect prob of step if reach end of node
            else:
                outputs.append(log_p[:, 0, :])
                sequences.append(selected)
            # # not allow to finish if statr with end token
            # elif len(outputs) == 0:
            #     state = MultiStepsRecoder(config=self.config, nodes=nodes)
            # stop if reach end of token or get all nodes
            if state.all_finished():
                break
        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def forward(self, features, adj_lists, nodes, adj_matrixs, return_pi=False):
        """
        embeddings torch.Size([batch_size, graph_size, embed_dim])
        """
        embeddings, recons_loss = self.encoder(features, adj_lists, nodes, adj_matrixs)
        # print('embeddings', embeddings[0].mean(dim=-1))
        for feature, adj_list, node, embedding in zip(features, adj_lists, nodes, embeddings):
            feature = feature.unsqueeze(0)
            node = node.unsqueeze(0)
            _log_p, pi = self.decoder(feature, node, embedding)
            # drop last (end of token) node
            pi = pi[:, :-1]
        # cost, mask = self.problem.get_costs(input, pi)
        """
        _log_p torch.Size([batch_size, seq_len, graph_size + 1])
        pi torch.Size([batch_size, seq_len])
        ll torch.Size([batch_size])
        """
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask=None)

        return ll, pi, recons_loss

    def get_multi_actions(self, features, adj_lists, nodes, adj_matrixs):
        """
        :param features: torch.Size([batch_size, graph_size, feature_num])
        :param adj_lists: dict
        :param nodes: torch.Size([batch_size, graph_size])
        :return:
            logprobs: torch.Size([batch_size])
            node_absolute_actions: torch.Size([batch_size, seq_len])
        """
        logprobs, node_relative_idxs, recons_loss = self.forward(features, adj_lists, nodes, adj_matrixs, return_pi=True)
        node_absolute_actions = []
        # reflect actions to absolute nodes indexes
        for a, n in zip(node_relative_idxs, nodes):
            node_absolute_actions.append(n[a].unsqueeze(0))
        node_absolute_actions = torch.cat(node_absolute_actions, dim=0)

        return logprobs, node_absolute_actions, recons_loss
