#!/usr/bin/env python3
"""This module contains the function list_all"""
import pymongo
from pymongo import MongoClient



def list_all(mongo_collection):
    """lists all documents in a collection

    mongo_collection will be the pymongo collection object
    """
    return list(mongo_collection.find())
