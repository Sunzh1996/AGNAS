from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Normal_Genotype = namedtuple('Normal_Genotype', 'normal normal_concat')
Reduce_Genotype = namedtuple('Reduce_Genotype', 'reduce reduce_concat')

PARAMS = {'conv_3x1_1x3':864, 'conv_7x1_1x7':2016, 'sep_conv_7x7': 1464, 'conv 3x3':1296, 'sep_conv_5x5': 888, 'sep_conv_3x3':504, 'dil_conv_5x5': 444, 'conv 1x1':144, 'dil_conv_3x3':252, 'skip_connect':0, 'none':0, 'max_pool_3x3':0, 'avg_pool_3x3':0}

PRIMITIVES = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'none'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V1



Cell_0 = Normal_Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6))
Cell_2 = Reduce_Genotype(reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
Cell_3 = Normal_Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6))
Cell_5 = Reduce_Genotype(reduce=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
Cell_6 = Normal_Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6))
AGNAS = [Cell_0, Cell_2, Cell_3, Cell_5, Cell_6]


