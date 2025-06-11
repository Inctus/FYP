from datasets.adult import AdultDataset
from datasets.acs_income import ACSIncomeDataset
from datasets.acs_travel import ACSTravelTimeDataset
from mechanisms.dpsgd import DPSGDHyperparameters, DPSGDMechanism
from mechanisms.sgd import SGDMechanism, BaseHyperparameters
from mechanisms.agt_ss import AGTHyperparameters, AGTMechanism
from models.mlp import BinaryClassificationMLP, MLPHyperparameters
from models.agt_mlp import AGTBCMLP, AGTBCMLPHyperparameters
from util.privacy import PrivacyBudget

print("Hello World!")

dataset = AdultDataset()

print(f"Loaded dataset with n_features: {dataset.n_features}")

model_hyperparams = AGTBCMLPHyperparameters(
    mlp_layers=[32, 128],
)

print("Model Hyperparameters set")

model_constructor = lambda: AGTBCMLP(
    n_features=dataset.n_features,
    hyperparameters=model_hyperparams
)

print("Created model constructor")

mechanism = AGTMechanism(
    model_constructor=model_constructor,
    dataset=dataset,
)

print("Initialised Mechanism")

hyperparams = AGTHyperparameters(
    learning_rate = 3.0,
    n_epochs = 4,
    batch_size = 45555,
    patience = 50,
    clip_gamma = 0.05,
    lr_min = 0.00020326181977670704,
    lr_decay = 0.5,
    momentum = 0.88,
)
 
print("Mechanism Hyperparameters set")

results = mechanism.train(hyperparams, "cuda:0")

print("Trained model had accuracy: ", results.accuracy)