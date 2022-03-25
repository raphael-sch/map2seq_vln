import torch
from torch import nn
import numpy as np
from utils import padding_idx


class BaseAgent:
    def __init__(self, opts, env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.env = env

    @staticmethod
    def _pad_seq(seqs):
        max_len = max([len(seq) for seq in seqs])
        seq_tensor = list()
        seq_lengths = list()
        for seq in seqs:
            instr_encoding_len = len(seq)
            seq = seq + [padding_idx] * (max_len - instr_encoding_len)
            seq_tensor.append(seq)
            seq_lengths.append(instr_encoding_len)
        seq_tensor = np.array(seq_tensor, dtype=np.int64)
        seq_lengths = np.array(seq_lengths, dtype=np.int64)
        return seq_tensor, seq_lengths

    def get_batch(self):
        """ Extract instructions from a list of observations and calculate corresponding seq lengths. """
        seqs = [item['instr_encoding'] for item in self.env.batch]

        seq_tensor, seq_lengths = self._pad_seq(seqs)

        seq_tensor = torch.from_numpy(seq_tensor).to(self.device)
        seq_lengths = torch.from_numpy(seq_lengths)
        return seq_tensor, seq_lengths


class OutdoorVlnAgent(BaseAgent):
    def __init__(self, opts, env, encoder, model):
        super(OutdoorVlnAgent, self).__init__(opts, env)
        self.instr_encoder = encoder
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)

    def rollout(self, is_test):
        trajs = self.env.reset()  # a batch of the first panoid for each route_panoids
        agent_actions = []
        batch_size = len(self.env.batch)

        seq, seq_lengths= self.get_batch()
        (text_enc_outputs, text_enc_lengths), (first_ht, first_ct) = self.instr_encoder(seq, seq_lengths)  # LSTM encoded hidden states for instructions, [batch_size, 1, 256]
        h_t = first_ht
        c_t = first_ct

        ended = np.zeros(batch_size, dtype=np.bool)
        ended = torch.from_numpy(ended).to(self.device)
        h2_t = None
        c2_t = None

        a = torch.ones(batch_size, 1, dtype=torch.int64, device=self.device)
        heading_changes = torch.zeros(batch_size, 1, 1, dtype=torch.float32, device=self.device)
        t = torch.tensor([-1], dtype=torch.int64, device=self.device)
        num_act_nav = [batch_size]
        loss = 0
        total_steps = [0]
        for step in range(self.opts.max_route_len):
            image_features = self.env.get_imgs()

            junction_types = self.env.get_junction_type()
            t = t + 1
            policy_output, (h_t, c_t), (h2_t, c2_t) = self.model(text_enc_outputs,
                                                                 text_enc_lengths,
                                                                 image_features,
                                                                 a,
                                                                 junction_types,
                                                                 heading_changes,
                                                                 h_t,
                                                                 c_t,
                                                                 t,
                                                                 h2_t=h2_t,
                                                                 c2_t=c2_t
                                                                 )
            if is_test:
                a, heading_changes = self.env.action_select(policy_output, ended, num_act_nav, trajs, total_steps)
                #agent_actions.append(a)
            else:
                gold_actions = self.env.get_gt_action()
                target_ = gold_actions.masked_fill(ended, value=torch.tensor(4))
                loss += self.criterion(policy_output, target_) * num_act_nav[0]
                heading_changes = self.env.env.action_step(gold_actions, ended, num_act_nav, trajs, total_steps)
                a = gold_actions.unsqueeze(1)
            if not num_act_nav[0]:
                break
        loss /= total_steps[0]
        if is_test:
            return None, trajs, agent_actions
        else:
            return loss, None, None
