from datasets.adult import AdultDataset
from mechanisms.dpsgd import DPSGDHyperparameters, DPSGDMechanism
from mechanisms.agt_ss import AGTHyperparameters, AGTMechanism
from models.mlp import BinaryClassificationMLP, MLPHyperparameters
from models.agt_mlp import AGTBCMLP
from util.privacy import PrivacyBudget

print("Hello World!")

dataset = AdultDataset()

print(f"Loaded dataset with n_features: {dataset.n_features}")

model_hyperparams = MLPHyperparameters(
    mlp_layers=[32, 128],
    p_dropout=0.1,
)

print("Model Hyperparameters set")

model_constructor = lambda: AGTBCMLP(
    n_features=dataset.n_features,
    hyperparameters=model_hyperparams
)

print("Created model constructor")

mechanism = AGTMechanism(model_constructor, dataset)

print("Initialised Mechanism")

hyperparams = AGTHyperparameters(
    learning_rate=0.03,
    n_epochs=5,
    batch_size=8192,
    patience=30,
    clip_gamma=0.1,
)
 
print("Mechanism Hyperparameters set")

results = mechanism.train(hyperparams, "cuda:1")

print("Trained model had accuracy: ", results.accuracy)