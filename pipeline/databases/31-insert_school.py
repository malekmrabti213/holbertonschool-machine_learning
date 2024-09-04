#!/usr/bin/env python3
"""this module includes the function insert_school"""


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection based on kwargs

    mongo_collection will be the pymongo collection object
    Returns the new _id
    """
    arguments = {}
    for key, value in kwargs.items():
        arguments[key] = value

    result = mongo_collection.insert_one(arguments)

    return result.inserted_id
