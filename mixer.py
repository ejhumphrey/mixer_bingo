from __future__ import print_function

import argparse
import json
import jsonschema
import logging
import numpy as np
import networkx as nx
import os
import pandas as pd
import random
import sys

logger = logging.getLogger(name=__file__)


def _load_schema():
    schema_file = os.path.join(os.path.dirname(__file__),
                               'participant_schema.json')
    return json.load(open(schema_file))

__SCHEMA__ = _load_schema()


def validate(participant_data):
    """Check that a number of records conforms to the expected format.

    Parameters
    ----------
    participant_data : array_like of dicts
        Collection of user records to validate.

    Returns
    -------
    is_valid : bool
        True if the provided data validates.
    """
    is_valid = True
    try:
        jsonschema.validate(participant_data, __SCHEMA__)
    except jsonschema.ValidationError as failed:
        logger.debug("Schema Validation Failed: {}".format(failed))
        is_valid = False
    return is_valid


def tokenize(records):
    """Create a token mapping from objects to integers.

    Parameters
    ----------
    records : array_like of iterables.
        Collection of nested arrays.

    Returns
    -------
    enum_map : dict
        Enumeration map of objects (any hashable) to tokens (int).
    """
    unique_items = set(i for row in records for i in row)
    unique_items = sorted(list(unique_items))
    return dict([(k, n) for n, k in enumerate(unique_items)])


def items_to_bitmap(records, enum_map=None):
    """Turn a collection of sparse items into a binary bitmap.

    Parameters
    ----------
    records : iterable of iterables, len=n
        Items to represent as a matrix.

    enum_map : dict, or None, len=k
        Token mapping items to ints; if None, one will be generated and
        returned.

    Returns
    -------
    bitmap : np.ndarray, shape=(n, k)
        Active items.

    enum_map : dict
        Mapping of items to integers, if one is not given.
    """
    return_mapping = False
    if enum_map is None:
        enum_map = tokenize(records)
        return_mapping = True

    bitmap = np.zeros([len(records), len(enum_map)], dtype=bool)
    for idx, row in enumerate(records):
        for i in row:
            bitmap[idx, enum_map[i]] = True
    return bitmap, enum_map if return_mapping else bitmap


def categorical_sample(pdf):
    """Randomly select a categorical index of a given PDF.

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    pdf = pdf / pdf.sum()
    return int(np.random.multinomial(1, pdf).nonzero()[0])


WEIGHTING_FUNCTIONS = {
    'l0': lambda x: float(np.sum(x) > 0),
    'l1': lambda x: float(np.sum(x)),
    'mean': lambda x: float(np.mean(x)),
    'null': 0.0,
    'euclidean': lambda x: np.sqrt(x),
    'norm_euclidean': lambda x: np.sqrt(x) / 3.0,
    'quadratic': lambda x: x,
    'norm_quadratic': lambda x: x / 9.0
}


def build_graph(records, forced_edges=None, null_edges=None,
                interest_func='l0', seniority_func='l0',
                combination_func=np.sum):
    """writeme

    Parameters
    ----------
    data: pd.DataFrame
        Loaded participant records

    forced_edges: np.ndarray, or None
        One-hot assignment matrix; no row or column can sum to more than one.

    null_edges: np.ndarray, or None
        Matches to set to zero.

    interest_func: str
        'l1', 'l0'

    seniority_func: str
        'l1', 'l0'

    combination_func: function
        Numpy functions, e.g. prod, sum, max.

    Returns
    -------
    graph : networkx.Graph
        Connected graph to be factored.
    """
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    interest_bitmap, interest_enum = items_to_bitmap(records.interests)

    # Coerce null / forced edges for datatype compliance.
    null_edges = ([] if null_edges is None
                  else [tuple(v) for v in null_edges])
    forced_edges = ([] if forced_edges is None
                    else [tuple(v) for v in forced_edges])

    graph = nx.Graph()
    for i, row_i in records.iterrows():
        for j, row_j in records.iterrows():
            # Skip self, shared affiliations, or same grouping
            skip_conditions = [i == j,
                               (i, j) in null_edges,
                               (j, i) in null_edges,
                               row_i.affiliation == row_j.affiliation]
            if any(skip_conditions):
                continue

            # Interest weighting
            interest_weight = WEIGHTING_FUNCTIONS[interest_func](
                interest_bitmap[i] * interest_bitmap[j])

            # Seniority weighting
            seniority_weight = WEIGHTING_FUNCTIONS[seniority_func](
                (row_i.seniority - row_j.seniority) ** 2.0)

            if (i, j) in forced_edges or (j, i) in forced_edges:
                weights = [2.0 ** 32]
            else:
                weights = [interest_weight, seniority_weight]
            graph.add_weighted_edges_from([(i, j, combination_func(weights))])

    return graph


def harmonic_mean(values):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    return np.power(np.prod(values), 1.0 / len(values))


def select_matches(records, k_matches=5, forced_edges=None, null_edges=None,
                   interest_func='l0', seniority_func='l0',
                   combination_func=np.sum, seed=None):
    """Pick affinity matches, and back-fill randomly if under-populated.

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    null_edges = ([] if null_edges is None
                  else [tuple(v) for v in null_edges])
    forced_edges = ([] if forced_edges is None
                    else [tuple(v) for v in forced_edges])

    matches = {i: set() for i in range(len(records))}

    for k in range(k_matches):
        graph = build_graph(
            records, null_edges=null_edges, forced_edges=forced_edges,
            seniority_func='quadratic', interest_func='mean',
            combination_func=np.mean)
        forced_edges = None
        links = nx.max_weight_matching(graph)
        for row, col in links.items():
            null_edges += (row, col)
            matches[row].add(col)

    catch_count = 0
    rng = np.random.RandomState(seed=seed)
    for row in matches:
        possible_matches = set(range(len(records)))
        possible_matches = possible_matches.difference(matches[row])
        while len(matches[row]) != k_matches:
            col = rng.choice(np.asarray(possible_matches))
            matches[row].add(col)
            null_edges += [(row, col)]
            catch_count += 1

    logger.debug("backfilled %d" % catch_count)
    return matches


def select_topic(row_a, row_b):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    topics_a = parse_interests(row_a[7])
    topics_b = parse_interests(row_b[7])

    topics = list(set(topics_a).intersection(set(topics_b)))
    if topics:
        return topics[categorical_sample(np.ones(len(topics)))]

TEXT_FMTS = [
    ("Find someone from %s.", 'affiliation'),
    ("Find someone currently located in %s.", 'country'),
    ("Find someone who is an expert on %s", 'topics'),
    ("Find someone in academia at the %s level", 'education')]


TEXT = [
    "Find someone who works in industry",
    "Introduce someone to someone else",
    "Help someone solve a square",
    "Find someone who plays an instrument.",
    "Find someone who has attended ISMIR for more than 5 years",
    "Find someone for which this is their first ISMIR"]


def generate_text(rows, target_idx, matches, num_outputs=24):
    outputs = []

    for match_idx in matches[target_idx]:
        outputs.append("Talk to %s" % rows[match_idx][1])

    outputs.extend(TEXT)
    categories = {
        'affiliation': get_affilations(rows),
        'education': get_education(rows),
        'topics': get_topics(rows),
        'country': get_countries(rows)
    }

    while len(outputs) < num_outputs:
        fmt, key = random.choice(TEXT_FMTS)
        value = random.choice(categories[key])
        outputs.append(fmt % value)

    return outputs


def make_card(name, contents, outfile):
    """writeme

    Parameters
    ----------
    x

    Returns
    -------
    y
    """
    tex_lines = []

    tex_lines.append(r'\documentclass[10pt, a4paper]{article}')
    tex_lines.append(r'\usepackage{tikz}')
    tex_lines.append(r'\usepackage{fullpage}')
    tex_lines.append(r'\usetikzlibrary{positioning,matrix}')
    tex_lines.append(r'\renewcommand*{\familydefault}{\sfdefault}')
    tex_lines.append(r'\usepackage{array}')

    tex_lines.append(r'\begin{document}')
    tex_lines.append(r'\pagestyle{empty}')
    tex_lines.append(r'\begin{center}')

    tex_lines.append(r'\Huge ISMIR 2014 Mixer Bingo\\')
    tex_lines.append(r"\bigskip \huge \emph{%s} \\" % name)
    tex_lines.append(r'\normalsize')
    tex_lines.append(r'')
    tex_lines.append(r'\bigskip')

    random.shuffle(contents)
    c = contents[0:12] + [r'FREE'] + contents[12:24]

    tex_lines.append(r'\begin{tikzpicture}')

    tex_lines.append(r"""\tikzset{square matrix/.style={
    matrix of nodes,
    column sep=-\pgflinewidth, row sep=-\pgflinewidth,
    nodes={draw,
      text height=#1/2-2.5em,
      text depth=#1/2+2.5em,
      text width=#1,
      align=center,
      inner sep=0pt
    },
  },
  square matrix/.default=3.2cm
}""")

    tex_lines.append(r'\matrix [square matrix]')
    tex_lines.append(r'(shi)')
    tex_lines.append(r'{')

    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[0], c[1], c[2], c[3], c[4]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[5], c[6], c[7], c[8], c[9]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[10], c[11], c[12], c[13], c[14]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[15], c[16], c[17], c[18], c[19]))
    tex_lines.append(
        r"%s & %s & %s & %s & %s\\" % (c[20], c[21], c[22], c[23], c[24]))
    tex_lines.append(r'};')

    tex_lines.append(r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-1-\i.north east) -- (shi-5-\i.south east);')
    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-1-\i.north west) -- (shi-5-\i.south west);')

    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-\i-1.north west) -- (shi-\i-5.north east);')
    tex_lines.append(
        r'\foreach \i in {1,2,3,4,5}')
    tex_lines.append(
        r'\draw[line width=2pt] (shi-\i-1.south west) -- (shi-\i-5.south east);')

    tex_lines.append(r'\end{tikzpicture}')
    tex_lines.append('')
    tex_lines.append(r'\pagebreak')
    tex_lines.append('')

    tex_lines.append(r'\end{center}')
    tex_lines.append(r'\end{document}')

    with open(outfile, 'w') as f:
        for line in tex_lines:
            f.write("%s\n" % line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a',
                        help='writeme')
    parser.add_argument('--b', type=str,
                        default='apple',
                        help='writeme')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to the console.')
    args = parser.parse_args()
    sys.exit(0)
