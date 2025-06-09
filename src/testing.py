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

dataset = ACSTravelTimeDataset()

print(f"Loaded dataset with n_features: {dataset.n_features}")

model_hyperparams = MLPHyperparameters(
    mlp_layers=[512, 128],
    p_dropout=0.5,
)

print("Model Hyperparameters set")

model_constructor = lambda: BinaryClassificationMLP(
    n_features=dataset.n_features,
    hyperparameters=model_hyperparams
)

print("Created model constructor")

mechanism = SGDMechanism(
    model_constructor=model_constructor,
    dataset=dataset,
)

print("Initialised Mechanism")

hyperparams = BaseHyperparameters(
    learning_rate=0.01,
    n_epochs=100,
    batch_size=256,
    patience=50,
)
 
print("Mechanism Hyperparameters set")

results = mechanism.train(hyperparams, "cuda:1")

print("Trained model had accuracy: ", results.accuracy)