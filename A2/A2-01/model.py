import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import predict_transform


#Activation Functions

class Mish(nn.Module):
    """
    Mish activation: x * tanh(softplus(x))
    Used in YOLOv4 instead of LeakyReLU.
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


#Placeholder Layers

class EmptyLayer(nn.Module):
    """Placeholder for route and shortcut layers."""
    def forward(self, x):
        return x


class DetectionLayer(nn.Module):
    """Stores anchors for a YOLO detection layer."""
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


#Config Parser

def parse_cfg(cfgfile):
    """
    Parse a Darknet .cfg file into a list of block dictionaries.
    """
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n')

    lines = [x.strip() for x in lines]
    lines = [x for x in lines if x and x[0] != '#']

    blocks = []
    block  = {}

    for line in lines:
        if line[0] == '[':
            if block:
                blocks.append(block)
            block = {'type': line[1:-1].strip()}
        else:
            key, val = line.split('=', 1)
            block[key.strip()] = val.strip()

    if block:
        blocks.append(block)

    return blocks


#Module Builder

def create_modules(blocks):
    """
    Build a ModuleList from parsed Darknet config blocks.
    Supports: convolutional, shortcut, upsample, route, yolo, maxpool
    Handles: leaky, mish activations
    """
    net_info       = blocks[0]
    module_list    = nn.ModuleList()
    prev_filters   = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #Convolutional
        if x['type'] == 'convolutional':
            activation     = x['activation']
            batch_normalize = int(x.get('batch_normalize', 0))
            filters        = int(x['filters'])
            kernel_size    = int(x['size'])
            stride         = int(x['stride'])
            pad            = (kernel_size - 1) // 2 if int(x.get('pad', 0)) else 0
            bias           = not batch_normalize

            module.add_module(f'conv_{index}',
                nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            )

            if batch_normalize:
                module.add_module(f'batch_norm_{index}',
                    nn.BatchNorm2d(filters)
                )

            if activation == 'leaky':
                module.add_module(f'leaky_{index}',
                    nn.LeakyReLU(0.1, inplace=True)
                )
            elif activation == 'mish':
                module.add_module(f'mish_{index}', Mish())
            # 'linear' = no activation

        # Shortcut (residual)
        elif x['type'] == 'shortcut':
            module.add_module(f'shortcut_{index}', EmptyLayer())
            filters = prev_filters

        # Upsample
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            module.add_module(f'upsample_{index}',
                nn.Upsample(scale_factor=stride, mode='nearest')
            )
            filters = prev_filters

        # Route
        elif x['type'] == 'route':
            layers = [int(a.strip()) for a in x['layers'].split(',')]
            # convert relative to absolute indices
            # relative indices are negative, absolute are positive
            layers = [index + l if l < 0 else l for l in layers]
            filters = sum(output_filters[l] for l in layers)
            module.add_module(f'route_{index}', EmptyLayer())

        # MaxPool (YOLOv4 SPP block) 
        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride      = int(x['stride'])
            # same padding to keep spatial size when stride=1
            if stride == 1:
                pad = kernel_size // 2
                module.add_module(f'maxpool_{index}',
                    nn.MaxPool2d(kernel_size, stride, padding=pad)
                )
            else:
                module.add_module(f'maxpool_{index}',
                    nn.MaxPool2d(kernel_size, stride)
                )
            filters = prev_filters

        # YOLO detection 
        elif x['type'] == 'yolo':
            mask        = [int(m) for m in x['mask'].split(',')]
            all_anchors = [int(a) for a in x['anchors'].split(',')]
            all_anchors = [(all_anchors[i], all_anchors[i+1])
                           for i in range(0, len(all_anchors), 2)]
            anchors     = [all_anchors[m] for m in mask]
            module.add_module(f'Detection_{index}', DetectionLayer(anchors))
            filters = prev_filters

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


#Darknet Model

class MyDarknet(nn.Module):
    """
    Unified Darknet model supporting YOLOv3 and YOLOv4.
    Key additions over the notebook version:
      - Mish activation (YOLOv4)
      - MaxPool support (YOLOv4 SPP block)
      - Route layer with 3+ inputs (YOLOv4)
      - Cleaner forward() with stored outputs dict
    """
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        self.blocks      = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, cuda=False):
        modules  = self.blocks[1:]
        outputs  = {}
        write    = False
        detections = None

        for i, module in enumerate(modules):
            module_type = module['type']

            #conv / upsample / maxpool
            if module_type in ('convolutional', 'upsample', 'maxpool'):
                x = self.module_list[i](x)
                
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x     = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'route':
                layers = [int(a.strip()) for a in module['layers'].split(',')]
                layers = [i + l if l < 0 else l for l in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                else:
                    x = torch.cat([outputs[l] for l in layers], dim=1)

            # yolo detection
            elif module_type == 'yolo':
                anchors    = self.module_list[i][0].anchors
                inp_dim    = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = predict_transform(x, inp_dim, anchors, num_classes, cuda)

                if not write:
                    detections = x
                    write      = True
                else:
                    detections = torch.cat((detections, x), dim=1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        """Load pretrained Darknet weights from binary file."""
        with open(weightfile, 'rb') as fp:
            header  = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen   = self.header[3]
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            if module_type != 'convolutional':
                continue

            model           = self.module_list[i]
            batch_normalize = int(self.blocks[i + 1].get('batch_normalize', 0))
            conv            = model[0]

            if batch_normalize:
                bn          = model[1]
                num_bn      = bn.bias.numel()

                bn_biases   = torch.from_numpy(weights[ptr:ptr+num_bn]); ptr += num_bn
                bn_weights  = torch.from_numpy(weights[ptr:ptr+num_bn]); ptr += num_bn
                bn_mean     = torch.from_numpy(weights[ptr:ptr+num_bn]); ptr += num_bn
                bn_var      = torch.from_numpy(weights[ptr:ptr+num_bn]); ptr += num_bn

                bn.bias.data.copy_(bn_biases.view_as(bn.bias.data))
                bn.weight.data.copy_(bn_weights.view_as(bn.weight.data))
                bn.running_mean.copy_(bn_mean.view_as(bn.running_mean))
                bn.running_var.copy_(bn_var.view_as(bn.running_var))
            else:
                num_biases  = conv.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases]); ptr += num_biases
                conv.bias.data.copy_(conv_biases.view_as(conv.bias.data))

            num_weights = conv.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]); ptr += num_weights
            conv.weight.data.copy_(conv_weights.view_as(conv.weight.data))