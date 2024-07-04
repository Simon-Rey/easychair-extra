import csv
import enum
import os

import time
from collections.abc import Collection, Iterable

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from mip import Model, BINARY, xsum, maximize, OptimizationStatus
from sklearn.cluster import KMeans, DBSCAN

from easychair_extra.read import read_committee, read_submission, authors_as_list
from easychair_extra.submission import bid_similarity, topic_similarity


class Presentation:
    def __init__(
        self,
        name,
        pairwise_scores=None,
        title=None,
        authors=None,
        duration=None,
        group_of=None,
    ):
        self.name = str(name)
        self.title = title
        self.authors = authors
        self.duration = duration
        self.group_of = group_of
        self.pairwise_scores = pairwise_scores

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Presentation):
            return self.name == other.name
        return self.name.__eq__(other)

    def __le__(self, other):
        if isinstance(other, Presentation):
            return self.name <= other.name
        return self.name.__le__(other)

    def __lt__(self, other):
        if isinstance(other, Presentation):
            return self.name < other.name
        return self.name.__lt__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


class Session:
    def __init__(self, name, max_duration=None, min_duration=None, presentations=None):
        self.name = name
        self.min_duration = min_duration
        self.max_duration = max_duration
        if presentations is None:
            presentations = []
        self.presentations = presentations

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Session):
            return self.name == other.name
        return self.name.__eq__(other)

    def __le__(self, other):
        if isinstance(other, Session):
            return self.name <= other.name
        return self.name.__le__(other)

    def __lt__(self, other):
        if isinstance(other, Session):
            return self.name < other.name
        return self.name.__lt__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


class TimeSlot:
    def __init__(self, name, sessions=None):
        self.name = name
        if sessions is None:
            sessions = []
        self.sessions = sessions

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name.__eq__(other)

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)


def avg(iterable):
    d = 0
    res = 0
    for x in iterable:
        res += x
        d += 1
    if d > 0:
        return res / d
    return 0


def cluster_presentations(
    presentations: Collection[Presentation], n_clusters: int, algorithm: str = "kmeans"
) -> list[list[Presentation]]:
    score_matrix = np.array(
        [
            [p1.pairwise_scores.get(p2, 0) for p1 in presentations]
            for p2 in presentations
        ]
    )
    distance_matrix = 1 - score_matrix
    np.fill_diagonal(distance_matrix, 0)

    if algorithm == "kmeans":
        os.environ["OMP_NUM_THREADS"] = "16"
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(distance_matrix)
        labels = kmeans.labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric="precomputed", eps=0.9, min_samples=2)
        dbscan.fit(distance_matrix)
        labels = dbscan.labels_
        n_clusters = max(labels)
    else:
        raise ValueError("The algorithm is unknown")

    clusters = [[] for _ in range(n_clusters)]
    for item, label in zip(presentations, labels):
        if label >= 0:
            clusters[label].append(item)
    return clusters


def iterative_merge_clustering(
    presentations: Collection[Presentation],
    merging_bound: int,
    n_cluster_ratio_min: float = 0,
    n_cluster_ratio_max: float = 1,
) -> set[Presentation]:
    current_pres = set(presentations)
    new_presentations = set()
    while True:
        global_merged_happened = False
        min_n_clusters = max(2, int(n_cluster_ratio_min * len(current_pres)))
        max_n_clusters = min(
            len(current_pres), int(n_cluster_ratio_max * len(current_pres))
        )
        for n_clusters in range(min_n_clusters, max_n_clusters):
            clusters = cluster_presentations(current_pres, n_clusters)
            local_merged_happened = False
            for c in clusters:
                if len(c) > 1 and sum(p.duration for p in c) <= merging_bound:
                    print(f"Merging {c}")
                    current_pres.difference_update(c)
                    new_group = []
                    for p in c:
                        if p.group_of:
                            new_group.extend(p.group_of)
                        else:
                            new_group.append(p)
                    merged_presentation = Presentation(
                        name=" - ".join(str(p.name) for p in new_group),
                        duration=sum(p.duration for p in new_group),
                        pairwise_scores={
                            p2: sum(p1.pairwise_scores[p2] for p1 in c) / len(c)
                            for p2 in current_pres.union(new_presentations)
                        },
                        group_of=new_group,
                    )
                    merged_presentation.pairwise_scores[merged_presentation] = sum(
                        p1.pairwise_scores[p2] for p1 in c for p2 in c
                    ) / len(c)
                    new_presentations.add(merged_presentation)
                    for p in current_pres.union(new_presentations):
                        p.pairwise_scores[merged_presentation] = (
                            merged_presentation.pairwise_scores[p]
                        )
                    print(f"\t...done, new len = {len(current_pres)}")
                    local_merged_happened = True
                    global_merged_happened = True
            if local_merged_happened:
                break
        if not global_merged_happened:
            if len(new_presentations) > 0:
                current_pres = current_pres.union(new_presentations)
                new_presentations = set()
                print(
                    f"Remixing the new presentations in: len is now {len(current_pres)}"
                )
            else:
                break
    return current_pres


def schedule_flow(
    presentations: Collection[Presentation], sessions: Collection[Session]
) -> dict[Session, list[Presentation]]:
    graph = nx.DiGraph()

    graph.add_node("source", demand=sum(p.duration for p in presentations))

    print("Building network flow")
    for session in sessions:
        graph.add_edge(session, "target", capacity=1)

    for presentation in presentations:
        graph.add_edge("source", f"{presentation}_in", capacity=1)
        graph.add_edge(f"{presentation}_in", f"{presentation}_out", capacity=1)
        for session in sessions:
            graph.add_edge(f"{presentation}_out", session, capacity=1)

        for presentation2 in presentations:
            score = presentation.pairwise_scores.get(presentation2, 0)
            if score > 0:
                graph.add_edge(
                    f"{presentation}_out",
                    f"{presentation2}_in",
                    capacity=1,
                    weight=-score,
                )
    print(f"Done, {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    flow = nx.max_flow_min_cost(graph, "source", "target")
    for source in flow:
        print(source)
        for target, value in flow[source].items():
            if value > 0:
                print(f"\t{target} ({value})")
    mincost = nx.cost_of_flow(graph, flow)
    print(f"Minimum cost of value {mincost} found")

    schedule = {session: [] for session in sessions}
    session_names = {session.name for session in sessions}
    for presentation in presentations:
        if flow["source"][f"{presentation}_in"] > 0.9:
            print(f"Flow for {presentation}")
            connected_presentations = [presentation]
            current_node = f"{presentation}_in"
            while True:
                next_node = ""
                for node, value in flow[current_node[:-2] + "out"].items():
                    if value > 0.9:
                        print(f"\tfound flow to {node}")
                        next_node = node
                print(f"\tnext node is {next_node}")
                if next_node in session_names:
                    break
                for p in presentations:
                    if p.name == next_node[:-3]:
                        connected_presentations.append(p)
                current_node = next_node
            print(f"For {next_node} with {connected_presentations}")
            schedule[next_node] = connected_presentations
    print(schedule)

    edge_colors = []
    for u, v in graph.edges():
        if flow[u][v] > 0:
            edge_colors.append("black")
        else:
            edge_colors.append("lightgrey")

    pos = nx.shell_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=700)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color="black")
    plt.show()
    return schedule


class Objectives(enum.Enum):
    MIN_SCORE = 1
    TOTAL_SC0RE = 2
    MIN_THEN_TOTAL_SCORE = 3


def egalitarian_schedule_ilp(
    presentations: Collection[Presentation],
    sessions: Collection[Session],
    max_run_time: int = 300,
    objective: Objectives = Objectives.MIN_SCORE,
    initial_solution=False,
) -> dict[Session, list[Presentation]]:
    start_time = time.time()
    m = Model()

    # Variable: Presentation is in session
    pres_to_session_vars = {}
    for presentation in presentations:
        vars_dict = {}
        for session in sessions:
            vars_dict[session] = m.add_var(
                f"x_{presentation}_{session}", var_type=BINARY
            )
        pres_to_session_vars[presentation] = vars_dict
    current_time = time.time()
    print(f"Added presentations vars in {current_time - start_time:.3f}s")

    # Constraint: All presentations are assigned to exactly one session
    for vars_dict in pres_to_session_vars.values():
        m += xsum(vars_dict.values()) == 1
    print(
        f"Added presentation to unique session constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Variable: Session is not empty
    sessions_non_empty_vars = {
        s: m.add_var(f"y_{s}", var_type=BINARY) for s in sessions
    }
    print(
        f"Added session non empty vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: Session not empty
    for vars_dict in pres_to_session_vars.values():
        for session, v in vars_dict.items():
            m += v <= sessions_non_empty_vars[session]
    print(
        f"Added session non empty constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: Duration of the session
    for session in sessions:
        m += (
            xsum(pres_to_session_vars[p][session] * p.duration for p in presentations)
            <= session.max_duration
        )
        m += (
            xsum(pres_to_session_vars[p][session] * p.duration for p in presentations)
            >= session.min_duration * sessions_non_empty_vars[session]
        )
    print(
        f"Added session duration constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Variable: Papers are together in a session
    pres_together_vars = {}
    for pres1 in presentations:
        pres_together_vars[pres1] = {}
        for pres2 in presentations:
            if pres1 != pres2:
                vars_dict = {}
                for session in sessions:
                    vars_dict[session] = m.add_var(
                        f"z_{pres1}_{pres2}_{session}", var_type=BINARY
                    )
                pres_together_vars[pres1][pres2] = vars_dict
    print(
        f"Added together vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    # Constraint: If together variables are 1 then papers are in session, and reciprocally
    for pres1, other_vars_dict in pres_together_vars.items():
        for pres2, session_vars in other_vars_dict.items():
            for session, v in session_vars.items():
                m += v <= pres_to_session_vars[pres1][session]
                m += v <= pres_to_session_vars[pres2][session]
                m += (
                    pres_to_session_vars[pres1][session]
                    + pres_to_session_vars[pres2][session]
                    <= 1 + v
                )
    print(
        f"Added together constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
    )
    current_time = time.time()

    if objective in [Objectives.MIN_SCORE, Objectives.MIN_THEN_TOTAL_SCORE]:
        # Variable: Min score of a session
        sessions_min_score_vars = {
            session: m.add_var(f"a_{session}") for session in sessions
        }
        print(
            f"Added session min score vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

        # Constraint: Min score of a session is min match
        for pres1, other_vars_dict in pres_together_vars.items():
            for pres2, session_vars_dict in other_vars_dict.items():
                for session, v in session_vars_dict.items():
                    m += (
                        sessions_min_score_vars[session]
                        <= pres_together_vars[pres1][pres2][session]
                        * pres1.pairwise_scores[pres2]
                        + (1 - pres_together_vars[pres1][pres2][session]) * 100
                    )
        for session, v in sessions_min_score_vars.items():
            m += v <= sessions_non_empty_vars[session] * 100
        print(
            f"Added session min score constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

    if objective in [Objectives.TOTAL_SC0RE, Objectives.MIN_THEN_TOTAL_SCORE]:
        # Variable: Total score of a session
        sessions_total_score_vars = {
            session: m.add_var(f"b_{session}") for session in sessions
        }
        print(
            f"Added session total score vars in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

        # Constraint: Total score of a session is total match
        for session, v in sessions_total_score_vars.items():
            m += v == xsum(
                pres_together_vars[pres1][pres2][session] * pres1.pairwise_scores[pres2]
                for pres1, other_vars_dict in pres_together_vars.items()
                for pres2 in other_vars_dict
            )
        print(
            f"Added session total score constraints in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
        current_time = time.time()

    if objective == Objectives.MIN_SCORE:
        m.objective = maximize(xsum(sessions_min_score_vars.values()))
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    elif objective == Objectives.TOTAL_SC0RE:
        m.objective = maximize(xsum(sessions_total_score_vars.values()))
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    elif objective == Objectives.MIN_THEN_TOTAL_SCORE:
        m.objective = maximize(
            100 * len(sessions) * xsum(sessions_min_score_vars.values())
            + xsum(sessions_total_score_vars.values())
        )
        print(
            f"Added objectives in {time.time() - start_time:.3f}s (+{time.time() - current_time:.3f}s)"
        )
    else:
        raise ValueError("Incorrect value for the objective.")

    if initial_solution:
        start = [
            (pres_to_session_vars[p][s], int(p in s.presentations))
            for s in sessions
            for p in presentations
        ]
        start += [
            (
                pres_together_vars[p1][p2][s],
                int(p1 in s.presentations and p2 in s.presentations),
            )
            for s in sessions
            for p1, vars_dict in pres_together_vars.items()
            for p2 in vars_dict
        ]
        m.start = start

    opt_status = m.optimize(max_seconds=max_run_time)

    if opt_status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        solution = {session: [] for session in sessions}
        for presentation, vars_dict in pres_to_session_vars.items():
            for session, v in vars_dict.items():
                if v.x >= 0.9:
                    solution[session].append(presentation)
        return solution


def sum_aggregator(values):
    return sum(values)


def avg_aggregator(values):
    return avg(values)


def best_five_avg_aggregator(values):
    return avg_aggregator(sorted(values, reverse=True)[:5])


def min_aggregator(values):
    if len(values) == 0:
        return 0
    return min(values)


def leximin_aggregator(values):
    res = []
    for v in values:
        if isinstance(v, Iterable):
            res.extend(v)
        else:
            res.append(v)
    return tuple(sorted(res))


def additive_leximin_aggregator(values):
    res = []
    for v in values:
        if isinstance(v, Iterable):
            for i, s in enumerate(v):
                if len(res) <= i:
                    res.append(s)
                else:
                    res[i] += s
        else:
            res.append(v)
    return tuple(sorted(res))


def presentation_collection_score(presentations, session_aggregator, presentation_aggregator):
    return session_aggregator([presentation_aggregator([p1.pairwise_scores[p2] for p2 in presentations]) for p1 in presentations])


def session_score(session, session_aggregator, presentation_aggregator):
    return presentation_collection_score(session.presentations, session_aggregator, presentation_aggregator)


def schedule_to_str(sessions: Collection[Session]):
    res = ""
    for s in sorted(sessions):
        for p in sorted(s.presentations):
            res += str(p)
    return res


def schedule_local_search(
    sessions: Collection[Session],
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
):
    print("Starting local search")
    name_to_session = {s.name: s for s in sessions}
    while True:
        current_sessions_scores = {s: session_score(s, session_aggregator, presentation_aggregator) for s in sessions}
        current_schedule_score = schedule_aggregator(current_sessions_scores.values())
        best_new_schedule_score = None
        arg_best_new_score = None
        for session in sessions:
            for session2 in sessions:
                if session != session2:
                    for presentation in session.presentations:
                        if sum(p.duration for p in session2.presentations) + presentation.duration <= session2.max_duration:
                            if len(session.presentations) == 1:
                                new_session_score = (0,)
                            else:
                                new_session_score = [sum(presentation.pairwise_scores[p2] for p2 in session.presentations if p2 != presentation) / sum(p2.duration for p2 in session.presentations if p2 != presentation)]
                                for p1 in session.presentations:
                                    new_session_score.append(
                                        sum(p1.pairwise_scores[p2] for p2 in session.presentations if p2 != presentation) / sum(
                                            p2.duration for p2 in
                                            session.presentations if p2 != presentation))
                                new_session_score = tuple(sorted(new_session_score))

                            sum_score = sum(presentation.pairwise_scores[p2] for p2 in session2.presentations)
                            sum_score += presentation.pairwise_scores[presentation]
                            new_session2_score = [sum_score / (sum(p2.duration for p2 in session2.presentations) + presentation.duration)]
                            for p1 in session2.presentations:
                                sum_score = sum(p1.pairwise_scores[p2] for p2 in session2.presentations)
                                sum_score += p1.pairwise_scores[presentation]
                                new_session2_score.append(
                                    sum_score / (sum(p2.duration for p2 in session2.presentations) + presentation.duration)
                                )
                            new_session2_score = tuple(sorted(new_session2_score))

                            new_schedule_scores = [score for s, score in current_sessions_scores.items()
                                                   if s not in [session, session2]]
                            new_schedule_scores.append(new_session_score)
                            new_schedule_scores.append(new_session2_score)

                            new_schedule_score = schedule_aggregator(new_schedule_scores)
                            if new_schedule_score > current_schedule_score:
                                if best_new_schedule_score is None or new_schedule_score > best_new_schedule_score:
                                    best_new_schedule_score = new_schedule_score
                                    arg_best_new_score = session, session2, presentation
        if arg_best_new_score is not None:
            session, session2, presentation = arg_best_new_score
            name_to_session[session.name].presentations.remove(presentation)
            name_to_session[session2.name].presentations.append(presentation)
            print(f"I'm moving {presentation} from {session} to {session2}")
        else:
            print("No improvement, I'm stopping.")
            break


def greedy_schedule(
    sessions: Collection[Session],
    presentations: Collection[Presentation],
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
):
    current_pres = set(presentations)
    empty_session_names = [s.name for s in sessions]
    while len(current_pres) > 0:
        current_sessions_scores = {s: session_score(s, session_aggregator, presentation_aggregator) for s in sessions}
        best_new_schedule_score = None
        arg_best_new_score = None

        for presentation in current_pres:
            if len(empty_session_names) > 0:
                # If adding to any session is bad, might as well just open a new session.
                max_increase = max(
                    min_aggregator(presentation_collection_score(s.presentations + [presentation],
                                                  session_aggregator,
                                                  presentation_aggregator)) - min_aggregator(current_sessions_scores[s])
                    for s in sessions
                )
                if max_increase < 3:
                    session_name = empty_session_names.pop()
                    for s in sessions:
                        if s.name == session_name:
                            s.presentations.append(presentation)
                            current_pres.remove(presentation)
                            print(
                                f"Opened a new session {s} for {presentation} ({len(current_pres)} to go), new score: {best_new_schedule_score}"
                            )
                            break
                    break
            for session in sessions:
                if sum(p.duration for p in session.presentations) + presentation.duration <= session.max_duration:
                    new_session_score = [sum(presentation.pairwise_scores[p2] for p2 in session.presentations) / (sum(p2.duration for p2 in session.presentations) + presentation.duration)]
                    for p in session.presentations:
                        new_session_score.append(sum(p.pairwise_scores[p2] for p2 in session.presentations) / (sum(p2.duration for p2 in session.presentations) + presentation.duration))
                    new_session_score = tuple(sorted(new_session_score))
                    new_schedule_scores = [score for s, score in current_sessions_scores.items() if s != session]
                    new_schedule_scores.append(new_session_score)
                    new_schedule_score = schedule_aggregator(new_schedule_scores)
                    if best_new_schedule_score is None or new_schedule_score > best_new_schedule_score:
                        best_new_schedule_score = new_schedule_score
                        arg_best_new_score = session, presentation
        if arg_best_new_score is not None:
            s, p = arg_best_new_score
            s.presentations.append(p)
            current_pres.remove(p)
            print(
                f"Added {p} to {s} ({len(current_pres)} to go), new score: {best_new_schedule_score}"
            )


def schedule_to_csv(time_slots: Collection[TimeSlot], file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timeslot", "session", "submission #", "title"])

        for time_slot in time_slots:
            for session in time_slot.sessions:
                for presentation in session.presentations:
                    writer.writerow(
                        [
                            time_slot.name,
                            session.name,
                            presentation.name,
                            presentation.title,
                        ]
                    )


def schedule_to_html(
    time_slots: Collection[TimeSlot],
    file_path: str,
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            f"""<!DOCTYPE html>
<html lang="en"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://ecai2024.simonrey.fr/static/css/style.css">
    <title>ECAI 2024 Schedule</title>
</head>
<body>
<header><h1>ECAI-2024 — Schedule</h1></header>
<main>
<section><p>
<strong>Global Score</strong>: {schedule_aggregator([session_score(s, session_aggregator, presentation_aggregator) for s in [s for t in time_slots for s in t.sessions]])}</p>
<p><strong># sessions with min score < 5:</strong> {sum(1 for t in time_slots for s in t.sessions if session_score(s, min_aggregator, presentation_aggregator) < 5)}</strong></p>
<p><strong># papers with score < 10:</strong> {sum(1 for t in time_slots for s in t.sessions for p in s.presentations if presentation_aggregator([p.pairwise_scores[p2] for p2 in s.presentations]) < 10)}</strong></p>
<p><strong># papers with score < 5:</strong> {sum(1 for t in time_slots for s in t.sessions for p in s.presentations if presentation_aggregator([p.pairwise_scores[p2] for p2 in s.presentations]) < 5)}</strong></p></section>"""
        )
        for time_slot in time_slots:
            f.write(f"<section><div class='section-content'><h2>{time_slot.name}</h2>")
            for session in time_slot.sessions:
                presentations = session.presentations
                f.write(
                    f"<h3 style='margin-top: 30px'>Session {session.name} (min_score = {session_score(session, min_aggregator, presentation_aggregator)})</h3>"
                )
                f.write(
                    f"""<table class="lined-table center-margin"><tr><th>{len(presentations)} Paper{'s' if len(presentations) > 0 else ''}</th><th>Score</th></tr>"""
                )
                p_score = [(presentation, presentation_aggregator([presentation.pairwise_scores[p] for p in presentations])) for presentation in presentations]
                p_score.sort(key=lambda x: x[1], reverse=True)
                for presentation, score in p_score:
                    f.write(
                        f"<tr><td>#{presentation}: {presentation.title}</td><td>{score}</td></tr><tr style='border-top: none;'><td>"
                    )
                    for i, p in enumerate(presentations):
                        if i == len(presentations) - 1:
                            f.write(f"{p.name}: {presentation.pairwise_scores[p]}")
                        else:
                            f.write(f"{p.name}: {presentation.pairwise_scores[p]} &mdash;")
                    f.write(
                        f"</td><td></td></tr>"
                    )
                f.write("</table>")
            f.write("</div></section>")
        f.write("</main></body></html>")


def draw_similarities(similarities, num_bins):
    all_s = [s for sim_dict in similarities.values() for s in sim_dict.values()]
    bin_step = max(all_s) / num_bins
    bins_cutoff = [bin_step * k for k in range(num_bins + 1)]
    data = [0 for _ in bins_cutoff]
    for s in all_s:
        cutoff = 0
        while cutoff < num_bins and s >= bins_cutoff[cutoff]:
            cutoff += 1
        if cutoff > 0:
            data[cutoff - 1] += 1

    labels = [
        f"{round(bins_cutoff[i], 2)} - {round(bins_cutoff[i + 1], 2)}"
        for i in range(num_bins)
    ]
    labels.append(f"{round(bins_cutoff[-1], 2)} - {max(all_s)}")
    fig, ax = plt.subplots()
    ax.bar(labels, data)
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Similarity Bins")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarities")
    plt.tight_layout()
    plt.show()


def main():
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")

    committee = read_committee(
        os.path.join(csv_dir, "committee.csv"),
        bids_file_path=os.path.join(csv_dir, "bidding.csv"),
    )

    submissions = read_submission(
        os.path.join(csv_dir, "submission.csv"),
        review_file_path=os.path.join(csv_dir, "review.csv"),
        submission_topic_file_path=os.path.join(csv_dir, "submission_topic.csv"),
    )
    submissions["avg_total_scores"] = submissions.apply(
        lambda df_row: avg(df_row.get("total_scores", [])), axis=1
    )
    accepted_submissions = submissions[submissions["decision"] == "accept"]
    print("=" * 50)
    print(f"Data loaded: {len(accepted_submissions.index)} submissions")

    bid_level_weights = {"yes": 1, "maybe": 0.5}

    bid_sim = bid_similarity(accepted_submissions, committee, bid_level_weights)
    # draw_similarities(bid_sim, 10)
    topic_sim = topic_similarity(accepted_submissions)
    # draw_similarities(topic_sim, 10)
    aggregated_similarity = {}
    for sub_id1, similarities in bid_sim.items():
        aggregated_similarity[sub_id1] = {}
        for sub_id2, s in similarities.items():
            if sub_id1 == sub_id2:
                aggregated_similarity[sub_id1][sub_id2] = 0
            else:
                aggregated_similarity[sub_id1][sub_id2] = max(1, int(s * 1000))
                # s2 = topic_sim[sub_id1][sub_id2]
                # if s * s2 == 0 and s + s2 > 0:
                #     aggregated_similarity[sub_id1][sub_id2] = max(
                #         1, int(max(s, s2) * 1000)
                #     )
                # else:
                #     aggregated_similarity[sub_id1][sub_id2] = max(1, int(s * s2 * 10000))

    print("=" * 50)
    print("SIMILARITY SCORES")
    print(
        f"Number of 0 similarity: {len([s for d in aggregated_similarity.values() for s in d.values() if s == 0])}\n"
    )
    # draw_similarities(aggregated_similarity, 40)

    all_presentations = []
    accepted_submissions.apply(
        lambda df_row: all_presentations.append(
            Presentation(
                df_row["#"],
                title=df_row["title"],
                authors=authors_as_list(df_row["authors"]),
                pairwise_scores={
                    str(k): v for k, v in aggregated_similarity[df_row["#"]].items()
                },
                duration=1,
            )
        ),
        axis=1,
    )
    # all_presentations = all_presentations[:50]

    all_time_slots = [
        TimeSlot("Mon-am"),
        TimeSlot("Mon-pm"),
        TimeSlot("Tue-am"),
        TimeSlot("Tue-pm"),
        TimeSlot("Wed-am"),
        TimeSlot("Wed-pm"),
        TimeSlot("Thu-am"),
    ]

    for t in all_time_slots:
        for k in range(1, 11):
            t.sessions.append(Session(f"{t.name}_{k}", min_duration=5, max_duration=10))

    # schedule = schedule_flow(all_presentations[:8], [s for t in all_time_slots for s in t.sessions][:2])
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    # print(f"TOTAL: min={schedule_score_min(schedule.keys(), session_score_min, presentation_score_avg)}, total={schedule_score_sum(schedule.keys(), session_score_min, presentation_score_avg)}")
    # print(f"leximin={schedule_score_leximin(schedule.keys(), session_score_min, presentation_score_avg)}")

    # print("\n" + "=" * 50)
    # print("Merge clustering")
    # merged_presentations = iterative_merge_clustering(
    #     all_presentations, 6, n_cluster_ratio_min=0, n_cluster_ratio_max=0.1
    # )
    # assert sum(p.duration for p in merged_presentations) == sum(
    #     p.duration for p in all_presentations
    # )
    # print(f"Final number of presentations: {len(merged_presentations)}")
    # all_presentations = merged_presentations
    #
    # print("\n" + "=" * 50)
    # print("Cluster in 10")
    # clusters = cluster_presentations(all_presentations, 10, algorithm="kmeans")
    # for i, c in enumerate(clusters):
    #     print(f"Cluster {i}: {sum(p.duration for p in c)} {c}")
    #
    # print("\n" + "=" * 50)
    # print("MIN_SCORE ILP")
    # schedule = egalitarian_schedule_ilp(
    #     all_presentations,
    #     [s for t in all_time_slots for s in t.sessions],
    #     max_run_time=900,
    #     objective=Objectives.MIN_SCORE,
    # )
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    # print("\n" + "=" * 50)
    # print("MIN_THEN_TOTAL_SCORE ILP")
    # schedule = egalitarian_schedule_ilp(
    #     all_presentations,
    #     [s for t in all_time_slots for s in t.sessions],
    #     max_run_time=600,
    #     objective=Objectives.MIN_THEN_TOTAL_SCORE,
    #     initial_solution=True,
    # )
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    #
    # all_sessions = []
    # for t in all_time_slots:
    #     for session in t.sessions:
    #         new_presentations = []
    #         presentations_to_remove = []
    #         for presentation in session.presentations:
    #             if presentation.group_of:
    #                 new_presentations.extend(presentation.group_of)
    #                 presentations_to_remove.append(presentation)
    #         session.presentations.extend(new_presentations)
    #         for p in presentations_to_remove:
    #             session.presentations.remove(p)
    #         all_sessions.append(session)

    all_sessions = [s for t in all_time_slots for s in t.sessions]
    print("\n" + "=" * 50)
    print("Greedy Schedule")
    greedy_schedule(all_sessions, all_presentations)
    schedule_to_csv(all_time_slots, "greedy_schedule.csv")
    schedule_to_html(all_time_slots, "greedy_schedule.html")

    print("\n" + "=" * 50)
    print("Local search")
    schedule_local_search(all_sessions)

    print(
        f"TOTAL: min={min_aggregator([session_score(s, min_aggregator, avg_aggregator) for s in all_sessions])}, total={sum_aggregator([session_score(s, sum_aggregator, avg_aggregator) for s in all_sessions])}"
    )
    print(
        f"leximin={leximin_aggregator([session_score(s, leximin_aggregator, avg_aggregator) for s in all_sessions])}"
    )

    schedule_to_csv(all_time_slots, "final_schedule.csv")
    schedule_to_html(all_time_slots, "final_schedule.html")


if __name__ == "__main__":
    main()
