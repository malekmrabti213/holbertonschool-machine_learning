#!/usr/bin/env python3
"""this module contains the function update_topics"""


def update_topics(mongo_collection, name, topics):
    """changes all topics of a school document based on the name

    Args:
    - mongo_collection will be the pymongo collection object
    - name (string) will be the school name to update
    - topics (list of strings) will be the list of topics approached in the
      school
    """
    myquery = { "name": name }
    newvalues = { "$set": { "topics": topics } }

    mongo_collection.update_many(myquery, newvalues)