#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient


def top_students(mongo_collection):
    """ returns all students sorted by average score """
    pipe = [{"$unwind": "$topics"},
            {"$group":
                {"_id": '$_id',
                 "averageScore": {"$avg": "$topics.score"},
                 "name": {'$first': '$name'}
                 }
             },
            {"$sort": {"averageScore": -1}}
            ]
    agg = mongo_collection.aggregate(pipe)
    return agg
