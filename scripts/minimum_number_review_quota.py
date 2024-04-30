import os

from read import read_committee, read_submission
from reviewassignment import find_feasible_assignment, committee_to_bid_profile

if __name__ == "__main__":
    committee = read_committee(
        os.path.join("csv", "committee.csv"),
        bids_file_path=os.path.join("csv", "bidding.csv")
    )

    submissions = read_submission(os.path.join("csv", "submission.csv"))
    print(submissions)

    number_review_per_paper = 3

    reviewers = committee[committee["role"] == "PC member"]
    print(reviewers)

    bid_level_weights = {'yes': 1, 'maybe': 0.5}
    bid_profile = committee_to_bid_profile(reviewers, submissions, bid_level_weights)

    all_paper_ids = set(submissions["#"].values)
    all_papers_with_bids = set()
    for bids in bid_profile.values():
        all_papers_with_bids.update(bids["yes"])
        all_papers_with_bids.update(bids["maybe"])
    all_papers_without_bids = all_paper_ids - all_papers_with_bids

    print(f"There are {len(all_paper_ids)} papers and "
          f"{len(all_paper_ids) - len(all_papers_without_bids)} of them received bids (missing "
          f"{len(all_papers_without_bids)}).")
    print("The following papers have not received bids:")
    for p in all_paper_ids:
        if p not in all_papers_with_bids:
            print(f"\t{p} - {submissions.loc[submissions['#'] == p]['title'].iloc[0]}")

    print("\n" + "=" * 43 + "\n   Looking for the smallest review quota\n" + "=" * 43)
    reviewers_assignment = {}
    num_reviews_needed = len(submissions.index) * number_review_per_paper
    number_max_review_per_reviewer = 0
    num_assigned_previous = 0
    while sum(len(p) for p in reviewers_assignment.values()) < num_reviews_needed:
        number_max_review_per_reviewer += 1
        reviewers_assignment = find_feasible_assignment(
            bid_profile,
            bid_level_weights,
            number_max_review_per_reviewer,
            number_review_per_paper
        )
        if reviewers_assignment:
            num_assigned = sum(len(p) for p in reviewers_assignment.values())
            print(
                f"\tFOUND: Assignment with {number_review_per_paper} reviews per paper and a "
                f"maximum of {number_max_review_per_reviewer} reviews per reviewers: "
                f"{num_assigned} (+{num_assigned - num_assigned_previous}) reviews in total for "
                f"{len(submissions.index) * number_review_per_paper} needed (missing "
                f"{len(submissions.index) * number_review_per_paper - num_assigned}).")
            if num_assigned == num_assigned_previous:
                break
            num_assigned_previous = num_assigned
        else:
            print(f"\tProblem solving the ILP...")
            break
