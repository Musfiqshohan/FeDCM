from argparse import Namespace
from MyProjects.modularUtils.evaluate_distributions import getdoKey, generate_permutations

variables = [
    'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
    'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',
    'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25',
    'treatment', 'y_factual'
]


covariates = variables[0:-2]  # x1 to x25

# Initialize DAG with covariates as exogenous
observed_dag = {var: [] for var in covariates}

# Add treatment and y_factual with covariate parents
observed_dag["treatment"] = covariates
observed_dag["y_factual"] = covariates + ["treatment"]

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
    "intv": [{'key': 'treatment', 'val': 0},
             {'key': 'treatment', 'val': 1}],

    'Observed_DAG': observed_dag,

    'confounders': {
    },

    'dcm_dim': {
        'x1': 10, 'x2': 10, 'x3': 10, 'x4': 10, 'x5': 10, 'x6': 10,

        'x7': 2, 'x8': 2, 'x9': 2, 'x10': 2, 'x11': 2, 'x12': 2,
        'x13': 2, 'x14': 2, 'x15': 2, 'x16': 2, 'x17': 2, 'x18': 2,
        'x19': 2, 'x20': 2, 'x21': 2, 'x22': 2, 'x23': 2, 'x24': 2,
        'x25': 2,

        'treatment': 2,
        'y_factual': 10,
    },


    'hid_dims': [256, 256],
}

args = Namespace(**arguments_dict)

args.intv_list = [
    {
        "expr": "P(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25)",
        "obs": ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                'x21', 'x22', 'x23', 'x24', 'x25'],
        "inter_vars": []
    }
]
