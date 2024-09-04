#!/usr/bin/env python3
"""This module contains the function schools_by_topic"""


def schools_by_topic(mongo_collection, topic):
    """returns the list of school having a specific topic

    Args:
    - mongo_collection will be the pymongo collection object
    - topic (string) will be topic searched
    """
    result = mongo_collection.find({"topics": topic})
    return list(result)