from experiment.fairness_eval import get_fairness_results, HYPERPARAM_RESULTS

for hyperparam_result_path in HYPERPARAM_RESULTS.keys():
    get_fairness_results(hyperparam_result_path)
