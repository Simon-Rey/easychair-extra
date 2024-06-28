def papers_without_pc(committee_df, submission_df):
    def aux(row):
        if row["all_authors_students"]:
            return None
        return not any(a in committee_df["person #"].values for a in row["authors_id"])

    submission_df["no_author_pc"] = submission_df.apply(aux, axis=1)
