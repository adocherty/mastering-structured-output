import numpy as np
from scipy import stats
import argparse
import pandas as pd


def results_to_table(results_list):

    df_results = {}
    for name, ss_results in results_list.items():
        df_results[name] = pd.DataFrame.from_dict(
            {
                tuple(mname.split("_", maxsplit=1)): {
                    tname: ss_results[mname][tname]["valid"] * 100
                    for tname in ss_results[mname].keys()
                }
                for mname in ss_results.keys()
            },
            orient="index",
        )
    return pd.concat(df_results).reorder_levels([1, 2, 0], axis=0).sort_index(axis=0)


def compare_m_experiments(
    results_list,
    p_value=0.05,
    bonferroni=True,
    alternative="greater",
    print_only_passed=False,
):
    """For each model compare the two experiments statistically.
    By default a one-sided test is used.
    Bonferroni correction is applied.
    """
    # Get all model names and select the subset shared by all experiments
    all_models = [set(inner_dict.keys()) for inner_dict in results_list.values()]
    model_list = set.intersection(*all_models)

    # Bonferroni correction
    n_tests = len(model_list)
    if bonferroni == True:
        alpha = p_value / n_tests
    elif isinstance(bonferroni, int):
        alpha = p_value / bonferroni
    else:
        alpha = p_value

    hypothesis_tests = {}
    for model in model_list:

        contingency_table = {}
        for name, ss_results in results_list.items():
            num_true = 0
            num_total = 0

            for tname in ss_results[model].keys():
                num_data = len(ss_results[model][tname]["outputs"])
                num_true += ss_results[model][tname]["valid"] * num_data
                num_total += num_data

            contingency_table[name] = {
                "Passed": num_true,
                "Failed": num_total - num_true,
            }

        ct = pd.DataFrame.from_dict(contingency_table, orient="index")

        # Ensure ordering to match hypotheses
        # Columns are experiments
        # Rows are outcomes
        # Column marginals are constant
        ct_n = ct[["Passed", "Failed"]].to_numpy().T
        sf = stats.fisher_exact(ct_n, alternative=alternative)
        sb = stats.barnard_exact(ct_n, alternative=alternative)

        hypothesis_tests[model] = {
            "Fisher exact": sf.pvalue,
            "Barnard exact": sb.pvalue,
            "Outcome": sb.pvalue < alpha,
        }

        if not print_only_passed or (sb.pvalue < alpha):
            print(f"\n{model}")
            print(f"Fisher exact p-value = {sf.pvalue}")
            print(f"Barnard exact p-value = {sb.pvalue}")
        if sb.pvalue < alpha:
            print(f"Hypothesis test passed: {sb.pvalue:.3g} < {alpha:.3g}")

    return pd.DataFrame.from_dict(hypothesis_tests, orient="index")


def monte_carlo_ci(successes, trials, confidence=0.95, num_samples=10000):
    """
    Calculate confidence interval using Monte Carlo simulation.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)
        num_samples (int): Number of Monte Carlo samples to generate

    Returns:
        tuple: (observed_proportion, lower_bound, upper_bound)
    """
    # Observed proportion
    p_hat = successes / trials

    # Create a distribution of possible true proportions
    possible_p = np.linspace(0.00001, 0.99999, 1000)

    # Storage for the likelihood of each possible p
    likelihoods = np.zeros_like(possible_p)

    # Calculate likelihood for each possible p
    for i, p in enumerate(possible_p):
        likelihoods[i] = stats.binom.pmf(successes, trials, p)

    # Normalize likelihoods to get a posterior distribution (assuming uniform prior)
    posterior = likelihoods / np.sum(likelihoods)

    # Draw samples from the posterior
    posterior_samples = np.random.choice(possible_p, size=num_samples, p=posterior)

    # Sort samples to find confidence interval bounds
    posterior_samples.sort()

    # Calculate bounds
    alpha = 1 - confidence
    lower_idx = int(num_samples * (alpha / 2))
    upper_idx = int(num_samples * (1 - alpha / 2))

    # Get the bounds
    lower_bound = posterior_samples[lower_idx]
    upper_bound = posterior_samples[upper_idx]

    return p_hat, lower_bound, upper_bound


def normal_approximation_ci(successes, trials, confidence=0.95):
    """
    Calculate confidence interval using normal approximation (Wald method).

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)

    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    p_hat = successes / trials
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    # Standard error
    se = np.sqrt(p_hat * (1 - p_hat) / trials)

    # Confidence interval
    lower_bound = max(0, p_hat - z * se)
    upper_bound = min(1, p_hat + z * se)

    return p_hat, lower_bound, upper_bound


def wilson_score_ci(successes, trials, confidence=0.95):
    """
    Calculate confidence interval using Wilson score method.
    Better for small samples or extreme proportions.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)

    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    p_hat = successes / trials
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    # Wilson score interval calculation
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    adjusted_se = (
        np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
    )

    lower_bound = max(0, center - z * adjusted_se)
    upper_bound = min(1, center + z * adjusted_se)

    return p_hat, lower_bound, upper_bound


def agresti_coull_ci(successes, trials, confidence=0.95):
    """
    Calculate confidence interval using Agresti-Coull method.
    Adds pseudo-observations to stabilize the interval.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)

    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    # Add z²/2 successes and z²/2 failures
    n_tilde = trials + z**2
    p_tilde = (successes + z**2 / 2) / n_tilde

    # Standard error for adjusted values
    se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)

    # Confidence interval
    lower_bound = max(0, p_tilde - z * se_tilde)
    upper_bound = min(1, p_tilde + z * se_tilde)

    # Return original proportion but adjusted CI
    return successes / trials, lower_bound, upper_bound


def exact_binomial_ci(successes, trials, confidence=0.95):
    """
    Calculate confidence interval using the exact (Clopper-Pearson) method.
    Based on the binomial cumulative distribution function.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)

    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    alpha = 1 - confidence

    # For proportion=0 or proportion=1, handle separately to avoid division by zero
    if successes == 0:
        lower_bound = 0
    else:
        lower_bound = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)

    if successes == trials:
        upper_bound = 1
    else:
        upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return successes / trials, lower_bound, upper_bound


def select_best_method(successes, trials):
    """
    Select the most appropriate confidence interval method based on sample size and proportion.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials

    Returns:
        str: Name of the recommended method
    """
    p_hat = successes / trials

    # For very small samples or extreme proportions, use exact method
    if trials < 30 or min(successes, trials - successes) < 5:
        return "exact_binomial"

    # For extreme proportions but larger samples, use Wilson
    if p_hat < 0.1 or p_hat > 0.9:
        return "wilson_score"

    # For moderate sample sizes with non-extreme proportions, use Agresti-Coull
    if trials < 100:
        return "agresti_coull"

    # For large samples with non-extreme proportions, normal approximation is fine
    return "normal_approximation"


def summarize_results(successes, trials, confidence=0.95):
    """
    Summarize confidence interval results for all methods.

    Args:
        successes (int): Number of passes/successes
        trials (int): Total number of trials
        confidence (float): Confidence level (default: 0.95)

    Returns:
        str: Summary text
    """
    methods = {
        "Normal Approximation (Wald)": normal_approximation_ci,
        "Wilson Score": wilson_score_ci,
        "Agresti-Coull": agresti_coull_ci,
        "Exact (Clopper-Pearson)": exact_binomial_ci,
    }

    best_method = select_best_method(successes, trials)
    method_to_key = {
        "normal_approximation": "Normal Approximation (Wald)",
        "wilson_score": "Wilson Score",
        "agresti_coull": "Agresti-Coull",
        "exact_binomial": "Exact (Clopper-Pearson)",
    }

    recommended = method_to_key[best_method]

    # Calculate results
    results = {}
    p_hat = successes / trials

    for name, method in methods.items():
        _, lower, upper = method(successes, trials, confidence)
        width = upper - lower
        results[name] = (lower, upper, width)

    # Create summary text
    summary = [f"Results for {successes} passes in {trials} trials:"]
    summary.append(f"Observed proportion: {p_hat:.4f} ({successes}/{trials})")
    summary.append(f"\n{confidence*100:.0f}% Confidence Intervals:")

    for name, (lower, upper, width) in results.items():
        if name == recommended:
            summary.append(
                f"✓ {name}: [{lower:.4f}, {upper:.4f}] (width: {width:.4f}) RECOMMENDED"
            )
        else:
            summary.append(f"  {name}: [{lower:.4f}, {upper:.4f}] (width: {width:.4f})")

    # Add interpretation
    recommended_lower, recommended_upper, _ = results[recommended]
    summary.append(f"\nInterpretation:")
    summary.append(
        f"We are {confidence*100:.0f}% confident that the true proportion of passes"
    )
    summary.append(f"is between {recommended_lower:.4f} and {recommended_upper:.4f}.")

    # Add recommendation explanation
    summary.append(f"\nMethodology:")
    if best_method == "exact_binomial":
        summary.append(
            "The Exact (Clopper-Pearson) method is recommended for small sample sizes"
        )
        summary.append(
            "or extreme proportions. It's more conservative but guaranteed to provide"
        )
        summary.append("at least the nominal coverage probability.")
    elif best_method == "wilson_score":
        summary.append(
            "The Wilson Score method is recommended for moderate sample sizes with"
        )
        summary.append(
            "extreme proportions. It provides better coverage than the normal"
        )
        summary.append("approximation when proportions are close to 0 or 1.")
    elif best_method == "agresti_coull":
        summary.append(
            "The Agresti-Coull method is recommended for moderate sample sizes."
        )
        summary.append(
            "It uses a simple adjustment to the normal approximation to provide"
        )
        summary.append("better coverage without being overly conservative.")
    else:
        summary.append(
            "The Normal Approximation method is recommended for large sample sizes"
        )
        summary.append(
            "with proportions not close to 0 or 1. It's simple and performs well"
        )
        summary.append("when the sample size is large enough.")

    return "\n".join(summary)
