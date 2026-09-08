#!/usr/bin/env python3
"""Build preference indexes used by the custom-KG KGSF extension.

The input KG is a mapping ``head_node -> [(relation_id, tail_node), ...]``.
All emitted IDs are unified KG node IDs (the IDs used by ``uikg.pkl``), never
movie titles or external ReDial IDs.
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

# A liked person may be a director, writer, or actor.  The
# supplied UIKG exposes movie -> person relations for the first three
# categories below; add a producer relation here if your KG later gains it.
REL = {
    "Directed_by": 0,
    "Written_by" : 1,
    "Starring": 2,
    "Genre": 3,
    "Liked_Movie": 11,
    "Disliked_Movie": 12,
    "Seen_Movie": 13,
    "Liked_Genre": 14,
    "Disliked_Genre": 15,
    "Liked_Person": 16,
    "Director": 17,
    "Write": 18,
    "Actor": 19,
    "Genre_Has": 20,
}

# Add the numeric Produced_by relation here when it exists in your ontology.
# The relation map supplied in this task contains no producer relation ID.
PERSON_RELATIONS = (REL["Directed_by"], REL["Written_by"], REL["Starring"])


def load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump(value, path: Path):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def as_node_id(value, entity_to_id):
    """Resolve a node ID whether the supplied pickle stores IDs or names."""
    if isinstance(value, int):
        return value
    if value in entity_to_id:
        return entity_to_id[value]
    raise KeyError("Cannot resolve %r to a unified KG node ID" % (value,))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    data_dir = args.data_dir

    entity_to_id = load(data_dir / "entity2entityId.pkl")
    kg = load(data_dir / "uikg.pkl")
    raw_users = load(data_dir / "user_ids.pkl")
    raw_movies = load(data_dir / "movie_ids.pkl")
    user_to_entity = load(data_dir / "uikg_userId2userEntity.pkl")

    user_keys = raw_users.keys() if isinstance(raw_users, dict) else raw_users
    movie_values = raw_movies.values() if isinstance(raw_movies, dict) else raw_movies
    # user_nodes = sorted({as_node_id(user_to_entity[u], entity_to_id) for u in user_keys
    #                         if u in user_to_entity})
    user_nodes = sorted({as_node_id(user, entity_to_id) for user in user_keys})
    movie_nodes = sorted({as_node_id(movie, entity_to_id) for movie in movie_values})
    movie_set = set(movie_nodes)

    liked_movies = defaultdict(set)
    disliked_movies = defaultdict(set)
    seen_movies = defaultdict(set)
    liked_genres = defaultdict(set)
    disliked_genres = defaultdict(set)
    liked_people = defaultdict(set)
    movie_to_genres = defaultdict(set)
    movie_to_people = defaultdict(set)

    for head, neighbors in kg.items():
        for relation, tail in neighbors:
            if relation == REL["Liked_Movie"] and tail in movie_set:
                liked_movies[head].add(tail)
            elif relation == REL["Disliked_Movie"] and tail in movie_set:
                disliked_movies[head].add(tail)
            elif relation == REL["Seen_Movie"] and tail in movie_set:
                seen_movies[head].add(tail)
            elif relation == REL["Liked_Genre"]:
                liked_genres[head].add(tail)
            elif relation == REL["Disliked_Genre"]:
                disliked_genres[head].add(tail)
            elif relation == REL["Liked_Person"]:
                liked_people[head].add(tail)
            elif head in movie_set and relation in (REL["Genre"], REL["Genre_Has"]):
                movie_to_genres[head].add(tail)
            elif head in movie_set and relation in PERSON_RELATIONS:
                movie_to_people[head].add(tail)

    # Fill attribute preferences inferred from explicit liked/disliked movies.
    for user, movies in liked_movies.items():
        liked_genres[user].update(*(movie_to_genres[movie] for movie in movies))
        liked_people[user].update(*(movie_to_people[movie] for movie in movies))
    inferred_disliked_genres = defaultdict(set)
    for user, movies in disliked_movies.items():
        inferred_disliked_genres[user].update(*(movie_to_genres[movie] for movie in movies))
    for user, genres in inferred_disliked_genres.items():
        disliked_genres[user].update(genres)

    # Store ordinary dictionaries of sorted lists: compact, deterministic, and
    # easy to consume from model.py.
    convert = lambda mapping: {key: sorted(value) for key, value in mapping.items()}
    dump(convert(liked_movies), data_dir / "user_liked_movies.pkl")
    dump(convert(disliked_movies), data_dir / "user_disliked_movies.pkl")
    dump(convert(seen_movies), data_dir / "user_seen_movies.pkl")
    dump(convert(liked_genres), data_dir / "user_liked_genres.pkl")
    dump(convert(disliked_genres), data_dir / "user_disliked_genres.pkl")
    dump(convert(liked_people), data_dir / "user_liked_people.pkl")
    dump(convert(movie_to_genres), data_dir / "movie_to_genres.pkl")
    dump(convert(movie_to_people), data_dir / "movie_to_people.pkl")
    person_to_movies = defaultdict(set)
    genre_to_movies = defaultdict(set)
    for movie, people in movie_to_people.items():
        for person in people:
            person_to_movies[person].add(movie)
    for movie, genres in movie_to_genres.items():
        for genre in genres:
            genre_to_movies[genre].add(movie)
    dump(convert(person_to_movies), data_dir / "person_to_movies.pkl")
    dump(convert(genre_to_movies), data_dir / "genre_to_movies.pkl")
    dump({
        "user_entity_ids": user_nodes,
        "movie_entity_ids": movie_nodes,
        "n_nodes": max(entity_to_id.values()) + 1,
        "n_relations": max(REL.values()) + 1,
        "relations": REL,
    }, data_dir / "uikg_metadata.pkl")
    print("Wrote custom-KG indexes for %d users and %d movies." %
          (len(user_nodes), len(movie_nodes)))


if __name__ == "__main__":
    main()
