import csv
import enum
import os

import time
from collections import Counter
from collections.abc import Collection, Iterable
from itertools import combinations

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from mip import Model, BINARY, xsum, maximize, OptimizationStatus, minimize
from sklearn.cluster import KMeans, DBSCAN

from easychair_extra.read import read_committee, read_submission, authors_as_list, read_topic
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
        topics=None,
        areas=None
    ):
        self.name = str(name)
        self.title = title
        self.authors = authors
        self.duration = duration
        self.group_of = group_of
        self.pairwise_scores = pairwise_scores
        if topics is None:
            topics = []
        self.topics = topics
        if areas is None:
            areas = []
        self.areas = areas

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
    def __init__(self, name, title="", max_duration=None, min_duration=None, presentations=None):
        self.name = name
        self.title = title
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
        ],
        dtype=float,
    )
    score_matrix /= score_matrix.max()
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
                    print(
                        f"\t...done, new pres duration = {merged_presentation.duration}, new len = {len(current_pres)}"
                    )
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


def unmerge_presentations(presentations: list[Presentation]):
    new_presentations = []
    presentations_to_remove = []
    for presentation in presentations:
        if presentation.group_of:
            new_presentations.extend(presentation.group_of)
            presentations_to_remove.append(presentation)
    presentations.extend(new_presentations)
    for p in presentations_to_remove:
        presentations.remove(p)


def unmerge_sessions(sessions: Collection[Session]):
    for session in sessions:
        unmerge_presentations(session.presentations)


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


def worst_three_avg_aggregator(values):
    return avg_aggregator(sorted(values)[:3])


def min_aggregator(values):
    if len(values) == 0:
        return 0
    return min(values)


def leximin_aggregator(values):
    if any(isinstance(v, Iterable) for v in values):
        res = []
        for v in values:
            res.extend(v)
        return tuple(sorted(res))
    return tuple(sorted(values))


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


def presentation_score(presentation, presentations, presentation_aggregator):
    return presentation_aggregator(
        presentation.pairwise_scores[p2] for p2 in presentations
    )


def presentation_collection_score(
    presentations, session_aggregator, presentation_aggregator
):
    return session_aggregator(
        [
            presentation_score(p1, presentations, presentation_aggregator)
            for p1 in presentations
        ]
    )


def session_score(session, session_aggregator, presentation_aggregator):
    return presentation_collection_score(
        session.presentations, session_aggregator, presentation_aggregator
    )


def schedule_to_str(sessions: Collection[Session]):
    res = ""
    for s in sorted(sessions):
        for p in sorted(s.presentations):
            res += str(p)
    return res


def schedule_reassignment_local_search(
    time_slots: Collection[TimeSlot],
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
    continuous_save=False,
        fixed_presentations=None,
):
    sessions = [s for t in time_slots for s in t.sessions]
    print("Starting re-assignment local search")
    while True:
        current_sessions_scores = {
            s: session_score(s, session_aggregator, presentation_aggregator)
            for s in sessions
        }
        current_schedule_score = schedule_aggregator(current_sessions_scores.values())
        best_new_schedule_score = None
        arg_best_new_score = None
        for session, session2 in combinations(sessions, 2):
            for presentation in session.presentations:
                if fixed_presentations and presentation not in fixed_presentations:
                    if (
                        sum(p.duration for p in session2.presentations)
                        + presentation.duration
                        <= session2.max_duration
                    ):
                        if len(session.presentations) == 1:
                            new_session_score = (0,)
                        else:
                            all_presentations = [
                                p for p in session.presentations if p != presentation
                            ]
                            new_session_score = presentation_collection_score(
                                all_presentations,
                                session_aggregator,
                                presentation_aggregator,
                            )

                        all_presentations_2 = session2.presentations + [presentation]
                        new_session2_score = presentation_collection_score(
                            all_presentations_2, session_aggregator, presentation_aggregator
                        )

                        new_schedule_scores = [
                            score
                            for s, score in current_sessions_scores.items()
                            if s not in [session, session2]
                        ]
                        new_schedule_scores.append(new_session_score)
                        new_schedule_scores.append(new_session2_score)

                        new_schedule_score = schedule_aggregator(new_schedule_scores)
                        if new_schedule_score > current_schedule_score:
                            if (
                                best_new_schedule_score is None
                                or new_schedule_score > best_new_schedule_score
                            ):
                                best_new_schedule_score = new_schedule_score
                                arg_best_new_score = session, session2, presentation
        if arg_best_new_score is not None:
            session, session2, presentation = arg_best_new_score
            session.presentations.remove(presentation)
            session2.presentations.append(presentation)
            print(f"I'm moving {presentation} from {session} to {session2}")
            print(f"New score: {best_new_schedule_score}")
            if continuous_save:
                schedule_to_csv(time_slots, "ls_reassignment_schedule.csv")
                schedule_to_html(time_slots, "ls_reassignment_schedule.html")
        else:
            print("No improvement, I'm stopping.")
            break


def schedule_swap_local_search(
    time_slots: Collection[TimeSlot],
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
    continuous_save=False,
        fixed_presentations=None
):
    print("Starting swap local search")
    sessions = [s for t in time_slots for s in t.sessions]
    improvement_values = {
        (s, p, s2, p2): None
        for s in sessions
        for p in s.presentations
        for s2 in sessions
        for p2 in s2.presentations
        if s != s2
    }
    while True:
        current_sessions_scores = {
            s: session_score(s, session_aggregator, presentation_aggregator)
            for s in sessions
        }
        all_swaps = dict()
        for session, session2 in combinations(sessions, 2):
            for presentation in session.presentations:
                if fixed_presentations and presentation not in fixed_presentations:
                    for presentation2 in session2.presentations:
                        if fixed_presentations and presentation2 not in fixed_presentations:
                            if (
                                improvement_values.get(
                                    (session, presentation, session2, presentation2)
                                )
                                is None
                            ):
                                all_presentations = [
                                    p for p in session.presentations if p != presentation
                                ] + [presentation2]
                                all_presentations_2 = [
                                    p for p in session2.presentations if p != presentation2
                                ] + [presentation]
                                if (
                                    sum(p.duration for p in all_presentations)
                                    <= session.max_duration
                                ):
                                    if (
                                        sum(p.duration for p in all_presentations_2)
                                        <= session2.max_duration
                                    ):
                                        new_session_score = presentation_collection_score(
                                            all_presentations,
                                            session_aggregator,
                                            presentation_aggregator,
                                        )

                                        new_session2_score = presentation_collection_score(
                                            all_presentations_2,
                                            session_aggregator,
                                            presentation_aggregator,
                                        )

                                        new_schedule_scores = [
                                            score
                                            for s, score in current_sessions_scores.items()
                                            if s not in [session, session2]
                                        ]
                                        new_schedule_scores.append(new_session_score)
                                        new_schedule_scores.append(new_session2_score)

                                        new_schedule_score = schedule_aggregator(
                                            new_schedule_scores
                                        )
                                        improvement_values[
                                            (session, presentation, session2, presentation2)
                                        ] = new_schedule_score
        current_schedule_score = schedule_aggregator(current_sessions_scores.values())
        for k, v in improvement_values.items():
            if v is not None and v > current_schedule_score:
                all_swaps[k] = v
        if len(all_swaps) > 0:
            print(f"{len(all_swaps)} potential swaps")
            swapped_sessions = set()
            for swap, score in sorted(
                all_swaps.items(), key=lambda x: x[1], reverse=True
            ):
                session, presentation, session2, presentation2 = swap
                if session not in swapped_sessions and session2 not in swapped_sessions:
                    session.presentations.remove(presentation)
                    session2.presentations.append(presentation)
                    session2.presentations.remove(presentation2)
                    session.presentations.append(presentation2)
                    swapped_sessions.add(session)
                    swapped_sessions.add(session2)
                    print(
                        f"I'm swapping {presentation} (from {session}) and {presentation2} (from {session2})"
                    )
            for k in improvement_values.keys():
                if k[0] in swapped_sessions or k[2] in swapped_sessions:
                    improvement_values[k] = None
            if continuous_save:
                schedule_to_csv(time_slots, "ls_swap_schedule.csv")
                schedule_to_html(time_slots, "ls_swap_schedule.html")
        else:
            print("No improvement, I'm stopping.")
            break


def greedy_schedule(
    sessions: Collection[Session],
    presentations: Collection[Presentation],
    schedule_aggregator=leximin_aggregator,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
    min_increase_new_session=3,
    assign_only_unassigned=False,
):
    if assign_only_unassigned:
        assigned_presentations = {p for s in sessions for p in s.presentations}
        current_pres = {p for p in presentations if p not in assigned_presentations}
    else:
        current_pres = set(presentations)
    print(f"Starting greedy scheduling, {len(current_pres)} presentations to assign")
    empty_sessions = [s for s in sessions if len(s.presentations) == 0]
    while len(current_pres) > 0:
        current_sessions_scores = {
            s: session_score(s, session_aggregator, presentation_aggregator)
            for s in sessions
        }
        best_new_schedule_score = None
        arg_best_new_score = None
        quick_assign = False

        for presentation in current_pres:
            eligible_sessions = [
                s
                for s in sessions
                if sum(p.duration for p in s.presentations) + presentation.duration
                <= s.max_duration
            ]
            if len(eligible_sessions) > 0:
                if len(empty_sessions) > 0:
                    # If adding to any session is bad, might as well just open a new session.
                    max_increase = max(
                        min_aggregator(
                            presentation_collection_score(
                                s.presentations + [presentation],
                                session_aggregator,
                                presentation_aggregator,
                            )
                        )
                        - min_aggregator(current_sessions_scores[s])
                        for s in sessions
                    )
                    if max_increase < min_increase_new_session:
                        session = empty_sessions.pop()
                        assert len(session.presentations) == 0
                        session.presentations.append(presentation)
                        current_pres.remove(presentation)
                        quick_assign = True
                        print(
                            f"Opened a new session {session} for {presentation} ({len(current_pres)} to go), new score: {best_new_schedule_score}"
                        )
                        assert (
                            sum(p.duration for p in session.presentations)
                            <= session.max_duration
                        )
                        break
                for session in eligible_sessions:
                    all_presentations = session.presentations + [presentation]
                    new_session_score = presentation_collection_score(
                        all_presentations, session_aggregator, presentation_aggregator
                    )
                    new_schedule_scores = [
                        score
                        for s, score in current_sessions_scores.items()
                        if s != session
                    ]
                    new_schedule_scores.append(new_session_score)
                    new_schedule_score = schedule_aggregator(new_schedule_scores)
                    if (
                        best_new_schedule_score is None
                        or new_schedule_score > best_new_schedule_score
                    ):
                        best_new_schedule_score = new_schedule_score
                        arg_best_new_score = session, presentation
        if arg_best_new_score is not None:
            s, p = arg_best_new_score
            s.presentations.append(p)
            current_pres.remove(p)
            try:
                empty_sessions.remove(s)
            except ValueError:
                pass
            print(
                f"Added {p} to {s} ({len(current_pres)} to go), new score: {best_new_schedule_score}"
            )
            assert sum(p.duration for p in s.presentations) <= s.max_duration
        elif not quick_assign:
            print(
                "Something went wrong, I'm stuck here and cannot place the last presentations..."
            )
            break


def schedule_to_csv(time_slots: Collection[TimeSlot], file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timeslot", "session", "session_title", "submission #", "title"])

        for time_slot in time_slots:
            for session in time_slot.sessions:
                for presentation in session.presentations:
                    writer.writerow(
                        [
                            time_slot.name,
                            session.name,
                            session.title,
                            presentation.name,
                            presentation.title,
                        ]
                    )


def import_csv(time_slots: Collection[TimeSlot], presentations: Collection[Presentation], file_path):
    sessions_mapping = {s.name: s for t in time_slots for s in t.sessions}
    presentation_mapping = {p.name: p for p in presentations}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions_mapping[row["session"]].presentations.append(presentation_mapping[row["submission #"].strip()])
            if "session_title" in row:
                sessions_mapping[row["session"]].title = row["title"]


def schedule_to_html(
    time_slots: Collection[TimeSlot],
    file_path: str,
    session_aggregator=leximin_aggregator,
    presentation_aggregator=avg_aggregator,
        fixed_presentations=None,
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
<section>
<p><strong>All paper scores</strong>: {leximin_aggregator([session_score(s, leximin_aggregator, presentation_aggregator) for t in time_slots for s in t.sessions])}</p>
<p><strong>Average paper score:</strong> {avg_aggregator([presentation_score(p, s.presentations, presentation_aggregator) for t in time_slots for s in t.sessions for p in s.presentations])}</strong></p>
<p><strong>All min session scores</strong>: {leximin_aggregator([session_score(s, min_aggregator, presentation_aggregator) for t in time_slots for s in t.sessions])}</p>
<p><strong># sessions with min score < 50:</strong> {sum(1 for t in time_slots for s in t.sessions if session_score(s, min_aggregator, presentation_aggregator) < 50)}</strong></p>
<p><strong># papers with score < 50:</strong> {sum(1 for t in time_slots for s in t.sessions for p in s.presentations if presentation_aggregator([p.pairwise_scores[p2] for p2 in s.presentations]) < 50)}</strong></p>
</section>"""
        )
        for time_slot in time_slots:
            f.write(f"<section><div class='section-content'><h2>{time_slot.name}</h2>")
            for session in time_slot.sessions:
                presentations = session.presentations
                f.write(
                    f"<h3 style='margin-top: 30px'>{session.title if session.title else session.name}</h3>"
                )
                areas_count = Counter()
                for p in session.presentations:
                    areas_count.update(p.areas)
                f.write("<p><strong>All areas:</strong> ")
                for i, a in enumerate(sorted(areas_count.items(), key=lambda x: x[1], reverse=True)):
                    if i == len(areas_count) - 1:
                        f.write(f"{a[0] }: {a[1]}")
                    else:
                        f.write(f"{a[0]}: {a[1]}, ")
                f.write("</p>")
                topics_count = Counter()
                for p in session.presentations:
                    topics_count.update(p.topics)
                f.write("<p><strong>All topics:</strong> ")
                for i, t in enumerate(sorted(topics_count.items(), key=lambda x: x[1], reverse=True)):
                    if i == len(topics_count) - 1:
                        f.write(f"{t[0] }: {t[1]}")
                    else:
                        f.write(f"{t[0]}: {t[1]}, ")
                f.write("</p>")
                f.write("<p><strong>Representative topics:</strong> ")
                repr_topics = representative_topics(session)
                for i in range(len(repr_topics)):
                    t = repr_topics[i]
                    f.write(f"{t} ({topics_count[t]}){', ' if i < len(repr_topics) - 1 else ''}")
                f.write("</p>")
                f.write(
                    f"""<table class="lined-table center-margin"><tr><th></th><th>{len(presentations)} Paper{'s' if len(presentations) > 0 else ''}</th><th>Score</th><th>#0</th></tr>"""
                )
                p_score = [
                    (
                        presentation,
                        presentation_aggregator(
                            [presentation.pairwise_scores[p] for p in presentations]
                        ),
                    )
                    for presentation in presentations
                ]
                p_score.sort(key=lambda x: x[1], reverse=True)
                for presentation, score in p_score:
                    f.write(
                        f"<tr><td><strong>{presentation} {'(X)' if fixed_presentations and presentation in fixed_presentations else ''}</strong></td><td><strong>{presentation.title}</strong><br>{', '.join(presentation.authors)}<br>"
                    )
                    num_zeros = 0
                    for i, p in enumerate(
                        sorted(
                            presentations, key=lambda x: presentation.pairwise_scores[x]
                        )
                    ):
                        if p != presentation:
                            s = presentation.pairwise_scores[p]
                            if s == 0:
                                num_zeros += 1
                            if i == len(presentations) - 1:
                                f.write(f"{p.name}: {s}")
                            else:
                                f.write(
                                    f"{p.name}: {s} &mdash;"
                                )
                    f.write(f"</td><td>{score}</td><td>{num_zeros}</td></tr>")
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


def assign_session_titles(sessions: Collection[Session]):
    for session in sessions:
        if not session.title:
            area_count = Counter()
            for p in session.presentations:
                area_count.update(p.areas)
            best = None
            arg_best = None
            for a, c in area_count.items():
                if best is None or c > best:
                    best = c
                    arg_best = a
            session.title = arg_best


def representative_topics(session: Session):
    m = Model()
    topic_count = Counter()
    for p in session.presentations:
        topic_count.update(p.topics)
    topic_vars = {t: m.add_var(f"x_{t}", var_type=BINARY) for t in topic_count}
    for p in session.presentations:
        m += xsum(topic_vars[t] for t in p.topics) >= 1
    m.objective = minimize(xsum(topic_vars.values()))
    m.verbose = False
    m.optimize()
    res = []
    for t, v in topic_vars.items():
        if v.x > 0.9:
            res.append(t)
    return res


def main():
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")

    committee = read_committee(
        os.path.join(csv_dir, "committee.csv"),
        bids_file_path=os.path.join(csv_dir, "bidding.csv"),
    )

    areas_to_topics, topics_to_areas = read_topic(
        os.path.join(csv_dir, "topics.csv"),
    )
    submissions = read_submission(
        os.path.join(csv_dir, "submission.csv"),
        review_file_path=os.path.join(csv_dir, "review.csv"),
        submission_topic_file_path=os.path.join(csv_dir, "submission_topic.csv"),
        topics_to_areas=topics_to_areas
    )
    submissions["avg_total_scores"] = submissions.apply(
        lambda df_row: avg(df_row.get("total_scores", [])), axis=1
    )
    accepted_submissions = submissions[submissions["decision"] == "accept"]
    print("=" * 50)
    print(f"Data loaded: {len(accepted_submissions.index)} submissions")

    cancelled_papers = []
    with open(os.path.join(csv_dir, "cancelled_papers.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cancelled_papers.append(int(row["#"].strip()))
    print(f"{len(cancelled_papers)} cancelled papers found.")
    accepted_submissions = accepted_submissions[~accepted_submissions["#"].isin(cancelled_papers)]
    print(f"Final number of {len(accepted_submissions.index)} submissions")

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
                s = int(s * 1000)
                aggregated_similarity[sub_id1][sub_id2] = 0 if s <= 50 else s
                # s2 = topic_sim[sub_id1][sub_id2]
                # if s * s2 == 0 and s + s2 > 0:
                #     aggregated_similarity[sub_id1][sub_id2] = max(
                #         1, int(max(s, s2) * 1000)
                #     )
                # else:
                # aggregated_similarity[sub_id1][sub_id2] = max(1, int(s * s2 * 1000))

    print("=" * 50)
    print("Similarity scores")
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
                topics=df_row["topics"],
                areas=df_row["areas"]
            )
        ),
        axis=1,
    )
    # all_presentations = all_presentations[:100]

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
        for k in range(1, 10):
            t.sessions.append(Session(f"{t.name}_{k}", min_duration=5, max_duration=10))
    all_sessions = [s for t in all_time_slots for s in t.sessions]

    print("=" * 50)
    print("Importing fixed sessions")
    import_csv(all_time_slots, all_presentations, "handpicked_schedule.csv")
    schedule_to_html(all_time_slots, "handpicked_schedule.html")
    fixed_presentations = {p for t in all_time_slots for s in t.sessions for p in s.presentations}
    all_presentations = [p for p in all_presentations if p not in fixed_presentations]
    print(f"{len(fixed_presentations)} presentations already assigned")
    print(f"{len(all_presentations)} presentations to assign")

    # schedule = schedule_flow(all_presentations[:8], [s for t in all_time_slots for s in t.sessions][:2])
    # for session, presentations in schedule.items():
    #     session.presentations = presentations
    # print(f"TOTAL: min={schedule_score_min(schedule.keys(), session_score_min, presentation_score_avg)}, total={schedule_score_sum(schedule.keys(), session_score_min, presentation_score_avg)}")
    # print(f"leximin={schedule_score_leximin(schedule.keys(), session_score_min, presentation_score_avg)}")

    print("\n" + "=" * 50)
    print("Merge clustering")
    merged_presentations = iterative_merge_clustering(
        all_presentations, 8, n_cluster_ratio_min=0, n_cluster_ratio_max=0.1
    )
    for p1 in fixed_presentations:
        for p2 in merged_presentations:
            if p2.group_of:
                p1.pairwise_scores[p2] = sum(p1.pairwise_scores[p] for p in p2.group_of) / len(p2.group_of)
                p2.pairwise_scores[p1] = p1.pairwise_scores[p2]
    assert sum(p.duration for p in merged_presentations) == sum(
        p.duration for p in all_presentations
    )
    print(f"Final number of presentations: {len(merged_presentations)}")
    all_presentations = list(merged_presentations)

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

    # schedule_to_csv(all_time_slots, "ilp_schedule.csv")
    # schedule_to_html(all_time_slots, "ilp_schedule.html")

    p_aggregator = avg_aggregator

    print("\n" + "=" * 50)
    print("Greedy Schedule")
    greedy_schedule(
        all_sessions, all_presentations, presentation_aggregator=p_aggregator
    )
    unmerge_sessions(all_sessions)
    unmerge_presentations(all_presentations)
    greedy_schedule(
        all_sessions,
        all_presentations,
        assign_only_unassigned=True,
        presentation_aggregator=p_aggregator,
    )
    schedule_to_csv(all_time_slots, "greedy_schedule.csv")
    schedule_to_html(all_time_slots, "greedy_schedule.html", fixed_presentations=fixed_presentations)

    print("\n" + "=" * 50)
    print("Reassignment local search")
    schedule_reassignment_local_search(
        all_time_slots, presentation_aggregator=p_aggregator, fixed_presentations=fixed_presentations
    )
    schedule_to_csv(all_time_slots, "ls_reassignment_schedule.csv")
    schedule_to_html(all_time_slots, "ls_reassignment_schedule.html", fixed_presentations=fixed_presentations)

    print("\n" + "=" * 50)
    print("Swap local search")
    schedule_swap_local_search(
        all_time_slots, continuous_save=True, presentation_aggregator=p_aggregator, fixed_presentations=fixed_presentations
    )
    schedule_to_csv(all_time_slots, "ls_swap_schedule.csv")
    schedule_to_html(all_time_slots, "ls_swap_schedule.html", fixed_presentations=fixed_presentations)

    print("\n" + "=" * 50)
    print(
        f"TOTAL: min={min_aggregator([session_score(s, min_aggregator, avg_aggregator) for s in all_sessions])}, total={sum_aggregator([session_score(s, sum_aggregator, avg_aggregator) for s in all_sessions])}"
    )
    print(
        f"leximin={leximin_aggregator([session_score(s, leximin_aggregator, avg_aggregator) for s in all_sessions])}"
    )

    assign_session_titles(all_sessions)
    schedule_to_csv(all_time_slots, "final_schedule.csv")
    schedule_to_html(all_time_slots, "final_schedule.html", fixed_presentations=fixed_presentations)


if __name__ == "__main__":
    main()
