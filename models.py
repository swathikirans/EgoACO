from copy import deepcopy
import torch.nn as nn
from egoACO import EgoACO


class VideoModel(nn.Module):

    def __init__(self, 
                 num_class, 
                 num_segments,
                 base_model='resnet34',
                 consensus_type='egoACO',
                 dropout=0.8, 
                 mem_size=512, 
                 output_pooling_classes=300
                 ):

        super(VideoModel, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.dropout = dropout
        self.consensus_type = consensus_type
        self.mem_size = mem_size
        self.output_pooling_classes = output_pooling_classes
        self.base_model_name = base_model

        attention_weights, feature_dim, w = self._prepare_base_model(base_model)
        self.consensus = EgoACO(input_dim=feature_dim, 
                                mem_size=self.mem_size,
                                attention_weights=attention_weights,
                                output_pooling_classes=self.output_pooling_classes,
                                num_class=self.num_class,
                                dropout=self.dropout, 
                                w=w
                                )
    def _prepare_base_model(self, base_model):

        if "resnet34" in base_model:
            from torchvision.models.resnet import resnet34
            self.base_model = resnet34(weights="IMAGENET1K_V1")
            self.base_model.layer4_noun = deepcopy(self.base_model.layer4)
            self.base_model.layer4_context = deepcopy(self.base_model.layer4)
            attention_weights = deepcopy(self.base_model.fc)
            self.base_model.fc = nn.Identity()
            feature_dim = 512
            w = 7

            def custom_forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
        
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                x_layer4 = self.layer4(x)
                x_layer4_noun = self.layer4_noun(x)
                x_layer4_context = self.layer4_context(x)

                return x_layer4, x_layer4_noun, x_layer4_context

            self.base_model.forward = custom_forward.__get__(self.base_model) # patch the forward method

        return attention_weights, feature_dim, w

    def forward(self, input):
        bz, c, num_segments, W, H = input.shape
        input = input.permute(0, 2, 1, 3, 4).reshape(bz*num_segments, c, W, H)

        base_out, base_out_noun, base_out_context = self.base_model(input)
        w, h = base_out.size()[-2:]
        base_out = base_out.view(bz, self.num_segments, -1, w, h)
        base_out_noun = base_out_noun.view(bz, self.num_segments, -1, w, h)
        base_out_context = base_out_context.view(bz, self.num_segments, -1, w, h)
        output = self.consensus(base_out_noun, base_out_context, base_out)

        return output