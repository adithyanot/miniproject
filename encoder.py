import torch
from transformers import XLMModel

class XLM_Encoder(torch.nn.Module):
    def __init__(self, xlm_model_path, hidden_size):
        super(XLM_Encoder, self).__init__()
        self.xlm_model = XLMModel.from_pretrained(xlm_model_path)
        self.hidden_size = hidden_size
        
        # Feed-forward network to project features into the same vector space
        self.W_h = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.b_h = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.W_l = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.b_l = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        # Gated value to control the flow of each feature
        self.W_g = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # Extracting linguistic features using XLM model
        outputs = self.xlm_model(input_ids=input_ids, attention_mask=attention_mask)
        ling_features = outputs.last_hidden_state
        
        # Projecting features into the same vector space
        h_t = torch.tanh(self.W_h(ling_features) + self.b_h(ling_features))
        l_t = torch.tanh(self.W_l(ling_features) + self.b_l(ling_features))
        
        # Learning the gated value g to control the flow of each feature
        h_l_t = torch.cat((h_t, l_t), dim=2)
        g = self.sigmoid(self.W_g(h_l_t))
        
        # Final encoder representation
        fused_features = g * h_t + (1 - g) * l_t
        
        return fused_features
