from copy import deepcopy

from mip import xsum, maximize, OptimizationStatus, Model, BINARY, LinExpr


def committee_to_bid_profile(committee_df, submission_df, bid_levels):
    bid_profile = {}

    def apply_func(df_row):
        bids = {}
        for bid in bid_levels:
            bids[bid] = [p for p in df_row["bids_" + bid] if p in submission_df["#"].values]
        bid_profile[df_row["#"]] = bids
    committee_df.apply(apply_func, axis=1)

    return bid_profile


def construct_mip_variables_for_assignment(
        bid_profile,
        bid_weights
):
    m = Model()
    reviewers_vars = {}
    reviewers_used_vars = {}
    submissions_vars = {}
    submissions_covered_vars = {}

    for r, bids in bid_profile.items():
        reviewers_vars[r] = {}
        reviewers_used_vars[r] = m.add_var(name=f"y_{r}", var_type=BINARY)
        for bid_level, bid_weight in bid_weights.items():
            if bid_weight > 0:
                for s in bids.get(bid_level, []):
                    variable = m.add_var(name=f"x_{r}_{s}", var_type=BINARY)
                    reviewers_vars[r][s] = variable
                    if s in submissions_vars:
                        submissions_vars[s][r] = variable
                    else:
                        submissions_vars[s] = {r: variable}
    for s in submissions_vars:
        submissions_covered_vars[s] = m.add_var(name=f"z_{s}", var_type=BINARY)
    return m, reviewers_vars, reviewers_used_vars, submissions_vars, submissions_covered_vars


def find_feasible_assignment(
        bid_profile: dict,
        bid_weights: dict,
        max_num_reviews_asked: int,
        num_reviews_per_paper: int,
        min_num_reviewers: bool = False,
        verbose: bool = False
) -> dict:

    m, reviewers_vars, reviewers_used_vars, submissions_vars, submissions_covered_vars = construct_mip_variables_for_assignment(
        bid_profile,
        bid_weights
    )

    if max_num_reviews_asked is not None:
        for r, sub_vars in reviewers_vars.items():
            m += xsum(sub_vars.values()) <= max_num_reviews_asked
    for s, rev_vars in submissions_vars.items():
        m += xsum(rev_vars.values()) <= num_reviews_per_paper

    if min_num_reviewers:
        for r, sub_vars in reviewers_vars.items():
            for sub_var in sub_vars.values():
                m += reviewers_used_vars[r] >= sub_var

    objective = LinExpr()
    for r, bids in bid_profile.items():
        for bid_level, bid_weight in bid_weights.items():
            if bid_weight > 0:
                for s in bids.get(bid_level, []):
                    objective += reviewers_vars[r][s] * bid_weight

    if min_num_reviewers:
        objective *= (1 + num_reviews_per_paper) * len(bid_profile)  # big M
        objective -= xsum(reviewers_used_vars.values())
    m.objective = maximize(objective)

    m.verbose = verbose
    status = m.optimize(max_seconds=600)
    if verbose:
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(m.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    solution = None
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        solution = {}
        for r, sub_vars in reviewers_vars.items():
            solution[r] = []
            for s, var in sub_vars.items():
                if abs(var.x) > 1e-6:
                    solution[r].append(s)
    return solution


def find_emergency_reviewers(
    bid_profile: dict,
    bid_weights: dict,
    max_num_reviewers: int,
    verbose: bool = False
) -> dict:

    m, reviewers_vars, reviewers_used_vars, submissions_vars, submissions_covered_vars = construct_mip_variables_for_assignment(
        bid_profile,
        bid_weights
    )

    # for r, sub_vars in reviewers_vars.items():
    #     for sub_var in sub_vars.values():
    #         m += reviewers_used_vars[r] >= sub_var
    #
    # for s, rev_vars in submissions_vars.items():
    #     for rev_var in rev_vars.values():
    #         m += submissions_covered_vars[s] >= rev_var

    # min_cover_vars = {}
    # for s in submissions_vars:
    #     min_cover_vars[s] = m.add_var(name=f"l_{s}", var_type=BINARY)
    #
    # for s, rev_vars in submissions_vars.items():
    #     for r in rev_vars:
    #         m += min_cover_vars[s] >= reviewers_used_vars[r]
    #
    # for s, v in min_cover_vars.items():
    #     m += submissions_covered_vars[s] <= v

    m += xsum(reviewers_used_vars.values()) <= max_num_reviewers

    # objective = xsum(submissions_covered_vars.values()) + 1
    # objective *= sum(len(bids) for bids in bid_profile.values())
    objective = xsum(len(bid_profile[r]) * v for r, v in reviewers_used_vars.items())
    m.objective = maximize(objective)

    m.verbose = verbose
    status = m.optimize(max_seconds=600)
    if verbose:
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(m.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    solution = None

    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        solution = {}
        for r, v in reviewers_used_vars.items():
            if v.x > 1e-6:
                sol = []
                for bid_level, bids in bid_profile[r].items():
                    if bid_weights[bid_level] > 0:
                        sol.extend(bids)
                solution[r] = sol
    return solution
