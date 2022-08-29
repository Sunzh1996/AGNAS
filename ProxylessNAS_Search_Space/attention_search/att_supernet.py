from torch_blocks import *


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class AttentionModule(nn.Module):
    def __init__(self, inp, out, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, out, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)

        return y


class AttSuperBlock(nn.Module):
    def __init__(self, inp, oup, stride):
        super(AttSuperBlock, self).__init__()
        self._ops = nn.ModuleList()
        self.input_channel = inp
        self.output_channel = oup
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        for idx, key in enumerate(config.blocks_keys):
            op = blocks_dict[key](inp, oup, stride)
            op.idx = idx
            self._ops.append(op)
        self.attention_layer = AttentionModule(
            inp=oup*(len(config.blocks_keys)+1*self.use_res_connect),
            out=oup*(len(config.blocks_keys)+1*self.use_res_connect),
            reduction=16
        )

    def forward(self, x, return_attention=False):
        block_features = []
        for idx, op in enumerate(self._ops):
            block_features.append(op(x))
        if self.use_res_connect:
            block_features.append(x)
        attention_weights = self.attention_layer(torch.cat(block_features, dim=1))
        split_attention_weights = []
        out_features = 0
        for idx, _feature in enumerate(block_features):
            _feature_attention_weights = attention_weights[:, idx * self.output_channel:(idx + 1) * self.output_channel, :, :]
            split_attention_weights.append(_feature_attention_weights)
            if idx == 0:
                out_features = _feature * _feature_attention_weights
            else:
                out_features += _feature * _feature_attention_weights

        if return_attention:
            return out_features, split_attention_weights
        return out_features


class AttSuperNetwork(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(AttSuperNetwork, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [  # for GPU search
            # t, c, n, s
            [6, 32, 4, 2],
            [6, 56, 4, 2],
            [6, 112, 4, 2],
            [6, 128, 4, 1],
            [6, 256, 4, 2],
            [6, 432, 1, 1],
        ]
        # building first layer
        input_channel = int(40 * width_mult)
        self.last_channel = int(1728 * width_mult) if width_mult > 1.0 else 1728
        self.conv_bn = conv_bn(3, input_channel, 2)
        self.MBConv_ratio_1 = InvertedResidual(input_channel, int(24 * width_mult), 3, 1, 1, 1)
        input_channel = int(24 * width_mult)
        self.features = nn.ModuleList()
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            t = None
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(AttSuperBlock(input_channel, output_channel, s))
                else:
                    self.features.append(AttSuperBlock(input_channel, output_channel, 1))
                input_channel = output_channel
        self.unshare_weights = nn.ModuleList()
        # building last several layers
        self.conv_1x1_bn = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, n_class),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.affine = True

    def forward(self, x, return_attention=False):
        x = self.conv_bn(x)
        x = self.MBConv_ratio_1(x)
        attention_weights_list = []
        for i, att_super_block in enumerate(self.features):
            if return_attention:
                x, attention_weights = att_super_block(x, return_attention)
                attention_weights_list.append(attention_weights)
            else:
                x = att_super_block(x)
        x = self.conv_1x1_bn(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)

        if return_attention:
            return x, attention_weights_list
        return x

    def architecture(self):
        arch = []
        for feat in self.features:
            if feat.stride == 2:
                arch.append('reduce')
            else:
                arch.append('normal')
        return arch


if __name__ == '__main__':
    import numpy as np

    model = AttSuperNetwork()
    inputs_shape = (2, 3, 224, 224)
    inputs = torch.rand(size=inputs_shape)
    # res = model(inputs)
    # print(res.shape)
    # print('Params.: %f M' % (sum(_param.numel() for _param in model.parameters()) / 1e6))

    # load pretrained weights
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pretrained_path = './exp_log/checkpoint_epoch_120.pth.tar'
    pretrained_weights = torch.load(pretrained_path, map_location=device)
    model_weights_no_module = {}
    for key in pretrained_weights['state_dict']:
        if 'module' in key:
            weights_name = key[7:]
        else:
            weights_name = key
        model_weights_no_module[weights_name] = pretrained_weights['state_dict'][key]
    model.load_state_dict(model_weights_no_module)
    model.eval()

    res, attention_weights_list = model(inputs, return_attention=True)
    for _attention_weights in attention_weights_list:
        channel_dim = _attention_weights[0].shape[1]
        op_weights = [torch.sum(_weights).item()/channel_dim for _weights in _attention_weights]
        op_weights = np.array(op_weights)
        print(op_weights)
