import torch
from torch import nn


class ORAR(nn.Module):
    def __init__(self, opts, instr_encoder, image_features=None):
        super(ORAR, self).__init__()
        self.opts = opts
        self.instr_encoder = instr_encoder

        self.dropout = opts.config.dropout
        self.rnn_state_dropout = opts.config.rnn_state_dropout
        self.attn_dropout = opts.config.attn_dropout

        num_heads = opts.config.num_heads
        rnn1_input_size = 16

        if opts.config.use_image_features:
            img_feature_shape = image_features.get('feature_shape', None)

            if self.opts.config.img_feature_dropout > 0:
                self.img_feature_dropout = nn.Dropout(p=self.opts.config.img_feature_dropout)

            img_feature_size = img_feature_shape[-1]
            assert len(img_feature_shape) == 2
            img_feature_flatten_size = img_feature_shape[0] * img_feature_shape[1]

            if img_feature_flatten_size > 2000:
                img_lstm_input_size = 256
                self.linear_img = nn.Linear(img_feature_flatten_size, 512)
                self.img_dropout = nn.Dropout(p=self.dropout)
                self.linear_img_extra = nn.Linear(512, img_lstm_input_size)
                self.img_dropout_extra = nn.Dropout(p=self.dropout)
            else:
                img_lstm_input_size = 64
                self.linear_img = nn.Linear(img_feature_flatten_size, img_lstm_input_size)
                self.img_dropout = nn.Dropout(p=self.dropout)

            rnn1_input_size += img_lstm_input_size

        self.action_embed = nn.Embedding(4, 16)
        if self.opts.config.junction_type_embedding:
            print('use pano embedding')
            self.junction_type_embed = nn.Embedding(4, 16)

        if self.opts.config.junction_type_embedding:
            rnn1_input_size += 16

        if self.opts.config.heading_change:
            rnn1_input_size += 1

        self.rnn = nn.LSTM(input_size=rnn1_input_size,
                           hidden_size=256,
                           num_layers=1,
                           batch_first=True)
        self.rnn_state_h_dropout = nn.Dropout(p=self.dropout)
        self.rnn_state_c_dropout = nn.Dropout(p=self.rnn_state_dropout)

        rnn2_input_size = 256
        if self.opts.config.use_text_attention:
            self.text_attention_layer = nn.MultiheadAttention(embed_dim=256,
                                                              num_heads=num_heads,
                                                              dropout=self.attn_dropout,
                                                              kdim=self.instr_encoder.hidden_size * 2,
                                                              vdim=self.instr_encoder.hidden_size * 2)
            if self.opts.config.use_layer_norm:
                self.layer_norm_text_attention = nn.LayerNorm(256, eps=1e-6)
            rnn2_input_size += 256

        if opts.config.use_image_features and self.opts.config.use_image_attention:
            self.visual_attention_layer = nn.MultiheadAttention(embed_dim=256,
                                                                num_heads=num_heads,
                                                                dropout=self.attn_dropout,
                                                                kdim=img_feature_size,
                                                                vdim=img_feature_size)
            if self.opts.config.use_layer_norm:
                self.layer_norm_visual_attention = nn.LayerNorm(256, eps=1e-6)
            rnn2_input_size += 256

        self.time_embed = nn.Embedding(self.opts.max_route_len, 32)
        rnn2_input_size += 32

        if self.opts.config.second_rnn:
            print('use second rnn')
            self.rnn2 = nn.LSTM(input_size=rnn2_input_size,
                                hidden_size=256,
                                num_layers=1,
                                batch_first=True)
            self.rnn2_state_h_dropout = nn.Dropout(p=self.dropout)
            self.rnn2_state_c_dropout = nn.Dropout(p=self.rnn_state_dropout)
        else:
            print('use no second rnn')
            self.policy_extra = nn.Linear(rnn2_input_size, 256)
        self.policy = nn.Linear(256, 4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, text_enc_outputs, text_enc_lengths, image_features, a, junction_types, heading_changes, h_t, c_t, t, h2_t=None, c2_t=None):
        """
        :param x: [batch_size, 1, 256], encoded instruction
        :param I: [batch_size, 1, 100, 100], features
        :param a: [batch_size, 1], action
        :param p: [batch_size, 1], pano type (street segment, T-intersection, 4-way intersection, >4 neighbors)
        :param h_t: [1, batch_size, 256], hidden state in LSTM
        :param c_t: [1, batch_size, 256], memory in LSTM
        :param t:
        :return:
        """

        rnn_input = []

        if self.opts.config.use_image_features:
            if self.opts.config.img_feature_dropout > 0:
                image_features = self.img_feature_dropout(image_features)

            rnn_image_features = image_features.flatten(start_dim=1)
            rnn_image_features = self.linear_img(rnn_image_features)
            rnn_image_features = torch.sigmoid(rnn_image_features)
            rnn_image_features = self.img_dropout(rnn_image_features)
            if hasattr(self, 'linear_img_extra'):
                rnn_image_features = self.linear_img_extra(rnn_image_features)
                rnn_image_features = torch.sigmoid(rnn_image_features)
                rnn_image_features = self.img_dropout_extra(rnn_image_features)
            rnn_input.append(rnn_image_features.unsqueeze(1))

        action_embedding = self.action_embed(a)  # [batch_size, 1, 16]
        rnn_input.append(action_embedding)
        if self.opts.config.junction_type_embedding:
            junction_type_embedding = self.junction_type_embed(junction_types).unsqueeze(1)  # [batch_size, 1, 16]
            rnn_input.append(junction_type_embedding)
        if self.opts.config.heading_change:
            rnn_input.append(heading_changes)  # [batch_size, 1, 1]


        s_t = torch.cat(rnn_input, dim=2)
        if h_t is None and c_t is None:  # first timestep
            _, (h_t, c_t) = self.rnn(s_t)
        else:
            _, (h_t, c_t) = self.rnn(s_t, (h_t, c_t))
        h_t = self.rnn_state_h_dropout(h_t)
        c_t = self.rnn_state_c_dropout(c_t)
        trajectory_hidden_state = h_t


        rnn2_input = [trajectory_hidden_state.squeeze(0)]  # [batch_size, 256]

        if self.opts.config.use_text_attention:
            text_attn = self._text_attention(trajectory_hidden_state, text_enc_outputs, text_enc_lengths)  # [1, batch_size, 256]
            rnn2_input.append(text_attn.squeeze(0))  # [batch_size, 256]

        if self.opts.config.use_image_features and self.opts.config.use_image_attention:
            if self.opts.config.use_text_attention:
                image_attn = self._visual_attention(text_attn, image_features)  # [1, batch_size, 256]
            else:
                image_attn = self._visual_attention(trajectory_hidden_state, image_features)  # [1, batch_size, 256]
            rnn2_input.append(image_attn.squeeze(0))  # [batch_size, 256]

        t = self.time_embed(t)
        batch_size = text_enc_lengths.size(0)
        t_expand = torch.zeros(batch_size, 32).to(self.device)
        t_expand.copy_(t)
        rnn2_input.append(t_expand)  # [batch_size, 32]

        rnn2_input = torch.cat(rnn2_input, dim=1)  # [batch_size, 256 + 256 + 256 + 32]
        action_t, (h2_t, c2_t) = self._forward_policy(rnn2_input, h2_t, c2_t)
        return action_t, (h_t, c_t), (h2_t, c2_t)

    def _forward_policy(self, policy_input, h2_t, c2_t):
        if self.opts.config.second_rnn:
            if h2_t is not None and c2_t is not None:
                _, (h2_t, c2_t) = self.rnn2(policy_input.unsqueeze(1), (h2_t, c2_t))
            else:  # first timestep
                _, (h2_t, c2_t) = self.rnn2(policy_input.unsqueeze(1))
            h2_t = self.rnn2_state_h_dropout(h2_t)
            c2_t = self.rnn2_state_c_dropout(c2_t)
            output_rnn2 = h2_t.squeeze(0)
        else:
            output_rnn2 = self.policy_extra(policy_input)

        action_t = self.policy(output_rnn2)
        return action_t, (h2_t, c2_t)

    def _text_attention(self, trajectory_hidden_state, text_enc_outputs, text_enc_lengths):
        key_padding_mask = ~(torch.arange(text_enc_outputs.shape[0])[None, :] < text_enc_lengths[:, None])
        key_padding_mask = key_padding_mask.to(self.device)

        attn, attn_weights = self.text_attention_layer(query=trajectory_hidden_state,
                                                       key=text_enc_outputs,
                                                       value=text_enc_outputs,
                                                       key_padding_mask=key_padding_mask)
        if self.opts.config.use_layer_norm:
            attn = self.layer_norm_text_attention(attn)
        return attn

    def _visual_attention(self, text_attn, image_features):
        image_features = image_features.permute(1, 0, 2)
        attn, attn_weights = self.visual_attention_layer(query=text_attn,
                                                         key=image_features,
                                                         value=image_features)
        if self.opts.config.use_layer_norm:
            attn = self.layer_norm_visual_attention(attn)
        return attn
