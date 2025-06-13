conf = {
    # export PYTHONPATH="${PYTHONPATH}:/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM"
    ##################################### The most commonly used parameters
    # data name
    # "data_name":"adult",
    "data_name": "ihdp",

    # global epochs
    "global_epochs": 1000,

    # local epoch
    "local_epochs": 100,

    "parallel_transform": False,

    "batch_size": 128,


    # number of clients
    "num_parties": 1,

    "isFL": True,
    "exp_name": "ihdp_fedcm",

    "cov_from":"model", # for ihdp: use "model" if cov is high dim. Otherwise use "train_data" if clients use their own data cov.

    "gm_conf": [
        {
            "expr": "P(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25)",
            "out_indx": {
                "x1": [], "x2": [], "x3": [], "x4": [], "x5": [],
                "x6": [], "x7": [], "x8": [], "x9": [], "x10": [],
                "x11": [], "x12": [], "x13": [], "x14": [], "x15": [],
                "x16": [], "x17": [], "x18": [], "x19": [], "x20": [],
                "x21": [], "x22": [], "x23": [], "x24": [], "x25": []
            },
            "in_indx": {}
        },
        # {
        #     "expr": "P(treatment| x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25)",
        #     "out_indx": {"treatment": []},
        #     "in_indx": {
        #         "x1": [], "x2": [], "x3": [], "x4": [], "x5": [],
        #         "x6": [], "x7": [], "x8": [], "x9": [], "x10": [],
        #         "x11": [], "x12": [], "x13": [], "x14": [], "x15": [],
        #         "x16": [], "x17": [], "x18": [], "x19": [], "x20": [],
        #         "x21": [], "x22": [], "x23": [], "x24": [], "x25": []
        #     }
        # },
        {
            "expr": "P(y_factual|treatment, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25)",
            "out_indx": {"y_factual": []},
            "in_indx": {
                "x1": [], "x2": [], "x3": [], "x4": [], "x5": [],
                "x6": [], "x7": [], "x8": [], "x9": [], "x10": [],
                "x11": [], "x12": [], "x13": [], "x14": [], "x15": [],
                "x16": [], "x17": [], "x18": [], "x19": [], "x20": [],
                "x21": [], "x22": [], "x23": [], "x24": [], "x25": [],
                "treatment": []
            }
        }
    ],

    # trained generator and discriminator
    "saved_models": {
        "ihdp": ["./checkpoints/ihdp/gen_epochxxx.pth", "./checkpoints/ihdp/disc_epochxxx.pth"]
    },

    # parameter of Dirichlet distribution
    # beta is used to control the level of noniid
    # the lower the beta, the unbalance the global data
    "beta": 100,

    # number of labels
    "num_classes": {
        "clinical": 2,
        "adult": 2,
        "intrusion": 10,
        "covtype": 7,
        "tb": 2,
        "credit": 2,
        "frontdoor": 2,
        "ihdp": 27
    },

    # label column
    "label_column": {
        "clinical": "label",
        "credit": "Class",
        "covtype": "Cover_Type",
        "intrusion": "label",
        "tb": "Condition",
        "IRIS": "species",
        "body": "class",
        "adult": "income_bracket",
        "frontdoor": "Y",
        "ihdp": "y_factual"
    },

    # test data file
    "test_dataset": {
        "clinical": "./data/clinical/clinical_test.csv",
        "intrusion": "./data/intrusion/intrusion_test.csv",
        "adult": "./data/adult/adult_test.csv",
        "frontdoor": "/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/frontdoor/test_data.csv",
        "ihdp": "/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/ihdp/test_data.csv",
        "covtype": "./data/covtype/covtype_test.csv"
    },

    # training data file
    "train_dataset": {
        "clinical": "./data/clinical/clinical_train.csv",
        "intrusion": "./data/intrusion/intrusion_train.csv",
        "adult": "./data/adult/adult_train.csv",
        "frontdoor": "/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/frontdoor",
        "ihdp": "/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/ihdp",
        "covtype": "./data/covtype/covtype_train.csv"
    },

    # model generated synthetic data
    "syn_data": {
        "clinical": "./data/clinical/clinical_syn.csv",
        "intrusion": "./data/intrusion/intrusion_syn.csv",
        "adult": "./data/adult/adult_syn.csv",
        "frontdoor": "./data/frontdoor/frontdoor_syn.csv",
        "ihdp": "./data/ihdp/ihdp_syn.csv",
        "covtype": "./data/covtype/covtype_syn.csv",
    },

    # discrete columns
    "discrete_columns": {
        "adult": [
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native_country',
            'income_bracket'
        ],
        "frontdoor": ['Z', 'X', 'Y'],

        "ihdp": [
            'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
            'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
            'x22', 'x23', 'x24', 'x25',
            'treatment'
        ],

        "intrusion": ['protocol_type', 'service', 'flag', 'label'],
        "credit": ["Class"],
        "covtype": ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Soil_Type1', 'Soil_Type2',
                    'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
                    'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
                    'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
                    'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
                    'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
                    'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
                    'Soil_Type40', 'Cover_Type'],

        "clinical": ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "label"],
        "tb": ["Condition"]
    },
    ##########################################################

    # max clusters for GMM
    "max_clusters": 10,

    # is fedavg
    "is_init_avg": False,

    "gen_weight_decay": 1e-5,

    # generator learning rate
    "gen_lr": 2e-4,

    # generator
    "generator_dim": (256, 256),

    "local_discriminator_steps": 3,

    "dis_weight_decay": 1e-5,

    # discriminator learning rate
    "dis_lr": 2e-4,

    "discriminator_dim": (256, 256),

    # generator input dimension
    "latent_size": 128,
}

##config
