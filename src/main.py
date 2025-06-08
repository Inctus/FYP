from experiment.hyperparameter_tuner import HyperparameterTuner
from datasets import ACSTravelTimeDataset, AdultDataset, ACSIncomeDataset
from mechanisms import SGDMechanism, BaseHyperparameters
from models import BinaryClassificationMLP

dataset = ACSTravelTimeDataset()
print("Loaded dataset")

mechanism = SGDMechanism(
    model=BinaryClassificationMLP,
    dataset=dataset,
)
print("Initialised Mechanism")

results = mechanism.train(BaseHyperparameters(
    learning_rate=0.01,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    early_stopping_patience=3
), device="cpu")
print("Trained model")