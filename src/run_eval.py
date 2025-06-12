from experiment.fairness_eval import get_fairness_results, HYPERPARAM_RESULTS

for hyperparam_result_path in ["acs_income_dataset/acs_income_agt.json", "adult_dataset/adult_agt.json"]:
    get_fairness_results(hyperparam_result_path)
