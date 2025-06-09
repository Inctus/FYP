from datasets.adult import AdultDataset
from mechanisms.dpsgd import DPSGDHyperparameters, DPSGDMechanism
from models.mlp import BinaryClassificationMLP, MLPHyperparameters
from util.privacy import PrivacyBudget

print("Hello World!")

dataset = AdultDataset()

print(f"Loaded dataset with n_features: {dataset.n_features}")

model_hyperparams = MLPHyperparameters(
    mlp_layers=[32, 128],
    p_dropout=0.1,
)

print("Model Hyperparameters set")

model_constructor = lambda: BinaryClassificationMLP(
    n_features=dataset.n_features,
    hyperparameters=model_hyperparams
)

print("Created model constructor")

mechanism = DPSGDMechanism(model_constructor, dataset)

print("Initialised Mechanism")

hyperparams = DPSGDHyperparameters(
    learning_rate=0.03,
    n_epochs=30,
    batch_size=256,
    patience=30,
    max_grad_norm=1.2,
)
 
print("Mechanism Hyperparameters set")

results = mechanism.train(hyperparams, PrivacyBudget(1.0, 1e-5), "cuda:1")

print("Trained model had accuracy: ", results.accuracy)