
from argparse import Namespace

from MyProjects.modularUtils.evaluate_distributions import getdoKey, generate_permutations

arguments_dict = {
    "Temperature": 1,
    "ANNEAL_RATE": 0.000003,
    "NOISE_DIM": 64,
    "dcm_epochs": 100,
    "saved_path": "./save",
    "learning_rate": 2e-4,
    "LAMBDA_GP": 1,
    "num_samples": 5000,
    "batch_size": 128,
    "intv": [{'key':'Z', 'val':0},
             {'key': 'Z', 'val': 1}],

    "max_intv" : None,


    # 'Observed_DAG': {
    #         "Z": [],
    #         "X": ['Z'],
    #         "Y": ['X']},
    # 'confounders':{'U': ['Z', 'Y']},

    'Observed_DAG': {
        "Z": [],
        "W": ['Z'],
        "X": ['Z'],
        "Y": ['W','X']},
    'confounders': {'U1': ['Z', 'Y'], 'U2': ['Z', 'W']},

    'dcm_dim':{},
    'hid_dims':[256,256]



}



args = Namespace(**arguments_dict)

intv_list = [
    {"expr": "P(X|do(Z))", "obs": ['X'], "inter_vars": ['Z']}
]


args.intv_list= intv_list

