#!/usr/bin/env python3
"""This module includes a script that provides some stats about Nginx
   logs stored in MongoDB"""
from pymongo import MongoClient

if __name__ == "__main__":
    """
    Gets stats about Nginx logs stored in MongoDB
    """
    client = MongoClient('mongodb://127.0.0.1:27017')

    collection = client.logs.nginx
    count_documents = len(list(collection.find()))
    print('{} logs'.format(count_documents))

    print('Methods:')

    n_get = len(list(collection.find({"method": "GET"})))
    print('\tmethod GET: {}'.format(n_get))

    n_post = len(list(collection.find({"method": "POST"})))
    print('\tmethod POST: {}'.format(n_post))

    n_put = len(list(collection.find({"method": "PUT"})))
    print('\tmethod PUT: {}'.format(n_put))

    n_patch = len(list(collection.find({"method": "PATCH"})))
    print('\tmethod PATCH: {}'.format(n_patch))

    n_delete = len(list(collection.find({"method": "DELETE"})))
    print('\tmethod DELETE: {}'.format(n_delete))

    n_status = len(list(collection.find({"method": "GET", "path": "/status"})))
    print('{} status check'.format(n_status))