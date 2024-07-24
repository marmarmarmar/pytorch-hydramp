from torch import nn

import torch


class HydrAMPGRU(nn.Module):
    
    def __init__(
        self,
        units=66,
        input_units=66,
        output_len=25,
    ):
        super().__init__()
        self.output_len = output_len
        self.units = units
        self.input_units = input_units
        self.kernel = torch.nn.Parameter(torch.zeros(size=(input_units, units * 3)))
        self.recurrent_kernel = torch.nn.Parameter(torch.zeros(size=(units, units * 3)))
        self.bias = torch.nn.Parameter(torch.zeros(size=(units * 3,)))
        
    
    def cell_forward(self, inputs, state):
        h_tm1 = state

        input_bias = self.bias

        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs
        
        matrix_x = torch.matmul(inputs, self.kernel)
        matrix_x = matrix_x + input_bias

        x_z, x_r, x_h = torch.split(matrix_x, self.units, dim=-1)

        matrix_inner = torch.matmul(h_tm1, self.recurrent_kernel[:self.units*2])

        recurrent_z, recurrent_r, recurrent_h = torch.split(
            matrix_inner, self.units, dim=-1
        )
        
        z = torch.sigmoid(x_z + recurrent_z)
        r = torch.sigmoid(x_r + recurrent_r)
        
        recurrent_h = torch.matmul(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:])

        hh = torch.tanh(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        new_state = h
        return h, new_state
    
    def forward(self, input_, state=None):
        if input_ is None:
            input_ = torch.zeros((state.shape[0], self.input_units))
        if state is None:
            state = torch.zeros((input_.shape[0], self.units))
        current_output = input_
        current_state = state
        outputs = []
        for _ in range(self.output_len):
            current_output, current_state = self.cell_forward(
                current_output,
                current_state,
            )
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)
    
    def forward_on_sequence(self, input_, state=None):
        if input_ is None:
            input_ = torch.zeros((state.shape[0], self.input_units))
        if state is None:
            state = torch.zeros((input_.shape[0], self.units))
        current_state = state
        outputs = []
        for i in range(self.output_len):
            current_output, current_state = self.cell_forward(
                input_[:, i],
                current_state,
            )
            outputs.append(current_output)
        return torch.stack(outputs, dim=1)
    
    
class HydrAMPDecoderTORCH(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gru = HydrAMPGRU()
        self.lstm = torch.nn.LSTM(68, 100, batch_first=True)
        self.dense = torch.nn.Linear(102, 21)
        
    def forward(self, x, return_logits=True, gumbel_temperature=0.001):
        gru_output = self.gru(None, x)
        condition = x[:, -2:]
        condition_repeat = condition.unsqueeze(1).repeat(1, 25, 1)
        gru_output_with_condition = torch.concat([gru_output, condition_repeat], dim=-1)
        lstm_output = self.lstm(gru_output_with_condition)[0]
        lstm_output_with_condition = torch.concat([lstm_output, condition_repeat], dim=-1)
        dense_output = self.dense(lstm_output_with_condition)
        if return_logits:
            return dense_output
        return torch.nn.functional.gumbel_softmax(dense_output, tau=gumbel_temperature)
    
    
class HydrAMPEncoderTorch(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=21,
            embedding_dim=100,
        )
        self.gru1_f = HydrAMPGRU(input_units=100, units=128)
        self.gru1_r = HydrAMPGRU(input_units=100, units=128)
        self.gru2_f = HydrAMPGRU(input_units=256, units=128)
        self.gru2_r = HydrAMPGRU(input_units=256, units=128)
        self.mean_linear = torch.nn.Linear(256, 64)
        self.std_linear = torch.nn.Linear(256, 64)
        
    def forward(self, x):
        embeddings = self.embedding(x)
        gru1_f_output = self.gru1_f.forward_on_sequence(embeddings)
        gru1_r_output = self.gru1_r.forward_on_sequence(torch.flip(embeddings, (1,)))
        gru_1_output = torch.concat([gru1_f_output, torch.flip(gru1_r_output, (1,))], dim=-1)
        gru2_f_output = self.gru2_f.forward_on_sequence(gru_1_output)
        gru2_r_output = self.gru2_r.forward_on_sequence(torch.flip(gru_1_output, (1,)))
        gru_2_output = torch.concat([gru2_f_output[:, -1], gru2_r_output[:, -1]], dim=-1)
        mean = self.mean_linear(gru_2_output)
        std = self.std_linear(gru_2_output)
        return mean, std