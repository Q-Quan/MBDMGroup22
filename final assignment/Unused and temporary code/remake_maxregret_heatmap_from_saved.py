#!/usr/bin/env python
# coding: utf-8

# # Model-Based Decision Making
#
# This notebook contains the final code of Group 22's assignment for the course EPA1361.
#
# In this code, we explore scenarios for the studied policy problem (flood risk management for the IJssel river) using open exploration, and then apply directed search (MORDM) to find optimal results.
#
# ## Table of contents
# 1. [Exploration](#Exploration)
#     1. [Problem formulation 3](#Problem-formulation-3)
#     2. [Problem formulation 4](#Problem-formulation-4)
# 2. [Optimization](#Optimization)
#     1. [Convergence](#Convergence)
#     2. [Constraints and solutions](#Constraints-and-solutions)
#     3. [Uncertainty](#Uncertainty)
#     4. [Scenario discovery using PRIM](#Scenario-discovery-using-PRIM)
#     5. [Sensitivity analysis using extra trees](#Sensitivity-analysis-using-extra-trees)
#     6. [Robustness](#Robustness)
#
# _The items in this table of contents may only be clickable in the browser version of Jupyter Notebook._

# In[ ]:


# Settings
save_figures = True
results_path = "./MBDM_Final_Data/Results"
figures_path = "./MBDM_Final_Data/Figures"

# In[2]:


# All needed imports
## Standard packages
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from pathlib import Path
import shutil
from datetime import datetime
import json
import numpy as np
import collections
from copy import deepcopy
## EMA workbench!
from ema_workbench import (
    Model,
    Policy,
    Scenario,
    MultiprocessingEvaluator,
    HypervolumeMetric,
    ScalarOutcome,
    ema_logging,
    save_results
)
## EMA workbench - optimization
from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress, to_problem)
## EMA workbench - analysis
from ema_workbench.analysis import (
    parcoords,
    prim,
    dimensional_stacking,
    feature_scoring,
    scenario_discovery_util
)
## IJssel dike model
from problem_formulation import get_model_for_problem_formulation


if __name__ == '__main__':

    # In[3]:


    # Set up
    ## Set up logging
    ema_logging.log_to_stderr(ema_logging.INFO)



    # ## Optimization

    # Get the model
    dike_model, planning_steps = get_model_for_problem_formulation(2)

    optimization_result = pd.read_csv(results_path + "/optimization_result.csv")
    optimization_result_constrained = pd.read_csv(results_path + "/optimization_result_constrained.csv")
    optimization_policies = pd.read_csv(results_path + "/optimization_policies.csv")

    # ### Uncertainty
    #
    # Re-evaluate candidate solutions under uncertainty. Performing experiments with 1000 scenarios for each of the policy options

    # In[ ]:


    # Build list of policies
    policies_to_evaluate = []

    for i, policy in optimization_policies.iterrows():
        policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))


    # In[ ]:


    n_scenarios = 10000 # the amount of scenarios for each policy option
    with MultiprocessingEvaluator(dike_model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(n_scenarios, policies_to_evaluate)

    experiments


    # In[ ]:


    outcomes


    # ### Scenario discovery using PRIM

    # In[ ]:


    # Clean up data = policy parameters, only leaving outcomes etc.
    columns_to_drop = ['A.1_DikeIncrease 0','A.1_DikeIncrease 1','A.1_DikeIncrease 2','A.2_DikeIncrease 0','A.2_DikeIncrease 1','A.2_DikeIncrease 2','A.3_DikeIncrease 0','A.3_DikeIncrease 1','A.3_DikeIncrease 2','A.4_DikeIncrease 0','A.4_DikeIncrease 1','A.4_DikeIncrease 2','A.5_DikeIncrease 0','A.5_DikeIncrease 1','A.5_DikeIncrease 2', 'policy']
    columns_to_drop += ['0_RfR 0','0_RfR 1','0_RfR 2','1_RfR 0','1_RfR 1','1_RfR 2','2_RfR 0','2_RfR 1','2_RfR 2','3_RfR 0','3_RfR 1','3_RfR 2','4_RfR 0','4_RfR 1','4_RfR 2','EWS_DaysToThreat']

    cleaned_experiments = experiments.copy()
    cleaned_experiments.drop(columns_to_drop, axis=1, inplace=True)


    # In[ ]:
    # #### Maximum regret

    # In[ ]:


    # Code as in model answer for assignment 9
    def calculate_regret(data, best):
        return np.abs(best-data)


    # In[ ]:


    overall_regret = {}
    max_regret = {}
    for outcome in dike_model.outcomes:
        policy_column = experiments['policy']

        # create a DataFrame with all the relevant information
        # i.e., policy, scenario_id, and scores
        data = pd.DataFrame({
            outcome.name: outcomes[outcome.name],
            "policy": experiments['policy'],
            "scenario": experiments['scenario']
        })

        # reorient the data by indexing with policy and scenario id
        data = data.pivot(index='scenario', columns='policy')

        # flatten the resulting hierarchical index resulting from
        # pivoting, (might be a nicer solution possible)
        data.columns = data.columns.get_level_values(1)

        outcome_regret = (data.max(axis=1).values[:, np.newaxis] - data).abs()

        overall_regret[outcome.name] = outcome_regret
        max_regret[outcome.name] = outcome_regret.max()


    # In[ ]:


    # Create heatmap
    max_regret_df = pd.DataFrame(max_regret)
    plt.figure()
    sns.heatmap(max_regret_df/max_regret_df.max(), cmap='viridis', annot=True)
    if save_figures:
        plt.savefig(figures_path + f"/maxregret_heatmap.png", dpi=300)
    #plt.show()