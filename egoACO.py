import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_


class LSTA(nn.Module):
    def __init__(self, 
                 input_size, 
                 memory_size, 
                 attention_weights, 
                 output_pooling_classes=150, 
                 kernel_size=3,
                 stride=1, 
                 padding=1, 
                 w = 7
                 ):
        
        super(LSTA, self).__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention_weights = nn.Linear(attention_weights.in_features, attention_weights.out_features, bias=False)
        self.attention_weights.weight = nn.Parameter(attention_weights.weight)
        self.attention_bias = nn.Parameter(attention_weights.bias.view(attention_weights.out_features, 1, 1).repeat(1, w, w))

        self.output_pooling_clasifier = nn.Linear(memory_size, output_pooling_classes, bias=False)
        self.coupling_fc = nn.Linear(input_size, output_pooling_classes, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Attention params
        self.conv_i_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_f_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_a_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_a_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_o_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        nn.init.xavier_normal_(self.conv_i_s.weight)
        constant_(self.conv_i_s.bias, 0)
        nn.init.xavier_normal_(self.conv_i_cam.weight)

        nn.init.xavier_normal_(self.conv_f_s.weight)
        constant_(self.conv_f_s.bias, 0)
        nn.init.xavier_normal_(self.conv_f_cam.weight)

        nn.init.xavier_normal_(self.conv_a_s.weight)
        constant_(self.conv_a_s.bias, 0)
        nn.init.xavier_normal_(self.conv_a_cam.weight)

        nn.init.xavier_normal_(self.conv_o_s.weight)
        constant_(self.conv_o_s.bias, 0)
        nn.init.xavier_normal_(self.conv_o_cam.weight)

        # Memory params
        self.conv_i_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_f_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_c_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_o_x = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)


        nn.init.xavier_normal_(self.conv_i_x.weight)
        constant_(self.conv_i_x.bias, 0)
        nn.init.xavier_normal_(self.conv_i_c.weight)

        nn.init.xavier_normal_(self.conv_f_x.weight)
        constant_(self.conv_f_x.bias, 0)
        nn.init.xavier_normal_(self.conv_f_c.weight)

        nn.init.xavier_normal_(self.conv_c_x.weight)
        constant_(self.conv_c_x.bias, 0)
        nn.init.xavier_normal_(self.conv_c_c.weight)

        nn.init.xavier_normal_(self.conv_o_x.weight)
        constant_(self.conv_o_x.bias, 0)
        nn.init.xavier_normal_(self.conv_o_c.weight)


    def forward(self, x, state_att, state_inp):
        bz, feat_planes, w, h = x.size()
        # state_att = [a, s]
        # state_inp = [atanh(c), o]

        a_t_1 = state_att[0]
        s_t_1 = state_att[1]

        c_t_1 = F.tanh(state_inp[0])
        o_t_1 = state_inp[1]

        # Attention recurrence
        inp_vector = self.avgpool(x).view(bz, feat_planes)
        feature_conv = x.view(bz, feat_planes, h * w)
        logits = self.attention_weights(inp_vector) + self.avgpool(self.attention_bias).view(self.attention_bias.size(0)).expand(bz, -1)
        probs, idxs = logits.sort(1, True)
        class_idx = idxs[:, 0]
        cam = torch.bmm(self.attention_weights.weight[class_idx].unsqueeze(1), feature_conv) + \
              self.attention_bias[class_idx].view(bz, 1, w*h)

        cam = cam.view(bz, 1, w, h)

        i_s = F.sigmoid(self.conv_i_s(s_t_1) + self.conv_i_cam(cam))
        f_s = F.sigmoid(self.conv_f_s(s_t_1) + self.conv_f_cam(cam))
        o_s = F.sigmoid(self.conv_o_s(s_t_1) + self.conv_o_cam(cam))
        a_tilde = F.tanh(self.conv_a_s(s_t_1) + self.conv_a_cam(cam))
        a = (f_s * a_t_1) + (i_s * a_tilde)
        s = o_s * F.tanh(a)
        u = s + cam  # hidden state + cam

        u = F.softmax(u.view(bz, w*h), 1)
        u = u.view(bz, 1, w, h)

        x_att = x * u.expand_as(x)

        i_x = F.sigmoid(self.conv_i_c(o_t_1 * c_t_1) + self.conv_i_x(x_att))
        f_x = F.sigmoid(self.conv_f_c(o_t_1 * c_t_1) + self.conv_f_x(x_att))
        c_tilde = F.tanh(self.conv_c_c(o_t_1 * c_t_1) + self.conv_c_x(x_att))
        c = (f_x * state_inp[0]) + (i_x * c_tilde)

        c_vec = self.avgpool(c).view(bz, self.memory_size)
        c_logits = self.output_pooling_clasifier(c_vec) + self.coupling_fc(self.avgpool(x_att).view(bz, feat_planes))
        c_probs, c_idxs = c_logits.sort(1, True)
        c_class_idx = c_idxs[:, 0]
        c_cam = self.output_pooling_clasifier.weight[c_class_idx].unsqueeze(2).unsqueeze(2) * c
        o_x = F.sigmoid(self.conv_o_x(o_t_1 * c_t_1) + self.conv_o_c(c_cam))

        state_att = [a, s]
        state_inp = [c, o_x]
        return state_att, state_inp, logits


class EgoACO(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 mem_size, 
                 attention_weights, 
                 output_pooling_classes, 
                 num_class, 
                 dropout, 
                 w):
        
        super(EgoACO, self).__init__()
        self.input_dim = input_dim
        self.mem_size = mem_size
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.attention_weights = attention_weights
        self.atten_dim = self.attention_weights.out_features

        self.attention_weights_noun = torch.nn.Linear(self.attention_weights.in_features, self.atten_dim, bias=False)
        self.attention_weights_noun.weight = torch.nn.Parameter(self.attention_weights.weight.clone())
        self.attention_weights_noun_bias = torch.nn.Parameter(self.attention_weights.bias.clone().view(attention_weights.out_features, 1, 1).repeat(1, w, w))

        self.attention_weights_context = torch.nn.Linear(self.attention_weights.in_features, self.atten_dim, bias=False)
        self.attention_weights_context.weight = torch.nn.Parameter(self.attention_weights.weight.clone())
        self.attention_weights_context_bias = torch.nn.Parameter(self.attention_weights.bias.clone().view(attention_weights.out_features, 1, 1).repeat(1, w, w))
        self.lsta = LSTA(
                        input_size=input_dim, 
                        memory_size=mem_size, 
                        attention_weights=attention_weights,
                        output_pooling_classes=output_pooling_classes
                        )

        self.verb_classifier = torch.nn.Linear(mem_size, num_class[0])
        self.noun_classifier = torch.nn.Linear(input_dim, num_class[1])
        self.action_classifier = torch.nn.Linear(mem_size+input_dim+input_dim, num_class[2])
        self.action_verb_bias = torch.nn.Linear(num_class[2], num_class[0])
        self.action_noun_bias = torch.nn.Linear(num_class[2], num_class[1])

        self.dropout_layer = torch.nn.Dropout(dropout)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        normal_(self.verb_classifier.weight, 0, 0.001)
        constant_(self.verb_classifier.bias, 0)
        constant_(self.noun_classifier.bias, 0)
        normal_(self.action_classifier.weight, 0, 0.001)
        constant_(self.action_classifier.bias, 0)
        normal_(self.action_verb_bias.weight, 0, 0.001)
        constant_(self.action_verb_bias.bias, 0)
        normal_(self.action_noun_bias.weight, 0, 0.001)
        constant_(self.action_noun_bias.bias, 0)

    def forward(self, x_noun, x_context, x):
        bz, time_steps, feat_planes, w, h = x.size()
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        x_noun = x_noun.permute(1, 0, 2, 3, 4).contiguous()
        x_context = x_context.permute(1, 0, 2, 3, 4).contiguous()

        x_noun_reshaped = x_noun.view(time_steps*bz, feat_planes, w, h)
        x_noun_reshaped1 = x_noun_reshaped.view(time_steps*bz, feat_planes, w*h)
        x_noun_avgPool = self.avgpool(x_noun_reshaped).view(time_steps*bz, feat_planes)

        x_context_reshaped = x_context.view(time_steps*bz, feat_planes, w, h)
        x_context_reshaped1 = x_context_reshaped.view(time_steps*bz, feat_planes, w*h)
        x_context_avgPool = self.avgpool(x_context_reshaped).view(time_steps*bz, feat_planes)

        x_logits_noun = self.attention_weights_noun(x_noun_avgPool) + self.avgpool(self.attention_weights_noun_bias).view(1, self.attention_weights_noun_bias.size(0)).expand(bz*time_steps, -1)
        x_logits = torch.mean(x_logits_noun.view(time_steps, bz, -1), dim=0, keepdim=False)
        x_logits = x_logits.unsqueeze(0).repeat(time_steps, 1, 1).view(time_steps*bz, -1)
        probs, idxs = x_logits.sort(1, True)
        class_idx = idxs[:, 0]
        cam_noun = torch.bmm(self.attention_weights_noun.weight[class_idx].unsqueeze(1), x_noun_reshaped1) + self.attention_weights_noun_bias[class_idx].view(bz*time_steps, 1, w*h)

        cam_noun = cam_noun.view(time_steps, bz, 1, w, h).permute(1, 2, 0, 3, 4).contiguous()
        noun_attention = self.softmax(cam_noun.view(bz, 1, time_steps*w*h)).view(bz, 1, time_steps, w, h).permute(2, 0, 1, 3, 4).contiguous()
        x_noun = x_noun_reshaped*noun_attention.view(time_steps*bz, 1, w, h)
        x_noun = self.avgpool(x_noun).view(time_steps*bz, feat_planes)
        noun_logits = self.noun_classifier(self.dropout_layer(x_noun)).view(time_steps, bz, -1)
        noun_logits = torch.mean(noun_logits, dim=0, keepdim=False)
        noun_feats = self.dropout_layer(torch.mean(x_noun.view(time_steps, bz, feat_planes), dim=0, keepdim=False))


        x_logits_context = self.attention_weights_context(x_context_avgPool)+ self.avgpool(self.attention_weights_context_bias).view(1, self.attention_weights_context_bias.size(0)).expand(bz*time_steps, -1)
        probs, idxs = x_logits_context.sort(1, True)
        class_idx = idxs[:, 0]
        cam_context = torch.bmm(self.attention_weights_context.weight[class_idx].unsqueeze(1), x_context_reshaped1) + self.attention_weights_context_bias[class_idx].view(bz*time_steps, 1, w*h)
        context_attention = self.softmax(cam_context.view(time_steps*bz, 1, w*h)).view(time_steps*bz, 1, w, h)
        x_context = x_context_reshaped*context_attention
        x_context = self.avgpool(x_context).view(time_steps*bz, feat_planes)
        x_context = self.dropout_layer(torch.mean(x_context.view(time_steps, bz, feat_planes), dim=0, keepdim=False))

        state_att = [torch.zeros([bz, 1, w, h], device=x.device, dtype=torch.float), torch.zeros([bz, 1, w, h], device=x.device, dtype=torch.float)]
        state_inp = [torch.zeros([bz, self.mem_size, w, h], device=x.device, dtype=torch.float), torch.zeros([bz, self.mem_size, w, h], device=x.device, dtype=torch.float)]
        for t in range(time_steps):
            state_att, state_inp, _ = self.lsta(x[t], state_att, state_inp)

        feats = self.avgpool(state_inp[0]).view(bz, self.mem_size)
        feats_logits = torch.cat([self.dropout_layer(feats), x_context, noun_feats], dim=1)
        action_logits = self.action_classifier(feats_logits)
        action_verb_bias = self.action_verb_bias(action_logits)
        action_noun_bias = self.action_noun_bias(action_logits)
        verb_logits = self.verb_classifier(self.dropout_layer(feats)) + action_verb_bias
        noun_logits = noun_logits + action_noun_bias
        return verb_logits, noun_logits, action_logits