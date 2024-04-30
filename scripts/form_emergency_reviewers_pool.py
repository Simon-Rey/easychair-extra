import warnings

import pandas as pd

import csv
import os.path

from read import read_committee, read_submission
from reviewassignment import find_feasible_assignment, committee_to_bid_profile


def read_volunteers(committee_df, emergency_pool_volunteers_file_path):

    all_full_names_with_count = committee_df['full name'].str.lower().value_counts()
    volunteers = []
    with open(emergency_pool_volunteers_file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            email = row["Email"]
            full_name = row["First Name"].lower().strip() + " " + row["Last Name"].lower().strip()
            full_name = full_name.replace('ł', 'l').replace('ń', 'n')
            if full_name in all_full_names_with_count and all_full_names_with_count[full_name] > 1:
                warnings.warn(f"There are more than one {full_name} in the committee, this "
                              f"information is not reliable. We drop it.")
                full_name = None
            volunteers.append((email, full_name))

    def is_volunteer(df_row):
        email_address = df_row["email"].lower()
        name = df_row["full name"].lower()
        names = (name, name.replace('ä', 'a'))
        for i, v in enumerate(volunteers):
            if email_address == v[0] or v[1] in names:
                del volunteers[i]
                return True
        return False

    committee_df["volunteer"] = committee_df.apply(is_volunteer, axis=1)

    if len(volunteers) > 0:
        new_line = "\n\t"
        warnings.warn(f"\n{len(volunteers)} volunteers could not be found in the committee, neither "
                      f"by name, nor by email address.\n\t{new_line.join(v[0] + ' ' + v[1] for v in volunteers)}")

    return committee_df


if __name__ == "__main__":
    committee = read_committee(
        os.path.join("csv", "committee.csv"),
        bids_file_path=os.path.join("csv", "bidding.csv")
    )

    submissions = read_submission(os.path.join("csv", "submission.csv"))

    print(submissions)

    read_volunteers(committee, os.path.join("csv", "emergency-pool-volunteers.csv"))

    volunteers = committee[committee["volunteer"]]

    bid_level_weights = {'yes': 1, 'maybe': 0.5}
    bid_profile = committee_to_bid_profile(volunteers, submissions, bid_level_weights)
    assignment = find_feasible_assignment(bid_profile, bid_level_weights, None, 1, min_num_reviewers=True)
    useful_reviewers = {r: a for r, a in assignment.items() if len(a) > 0}
    print()
    print(f"Found an assignment using {len(useful_reviewers)} reviewers:")
    for r, a in useful_reviewers.items():
        print(f"\t{r}: {a}")
    print(f"Total number of papers covered: {sum(len(a) for a in useful_reviewers.values())}")
