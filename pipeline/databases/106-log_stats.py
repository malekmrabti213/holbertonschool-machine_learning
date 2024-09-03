#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx = client.logs.nginx
    logs = nginx.count_documents({})
    # print(dir(logs))
    # print(logs.collection)
    ips = nginx.aggregate([
                          {'$group': {'_id': '$ip', 'sum': {'$sum': 1}}},
                          {"$sort": {"sum": -1}},
                          {"$limit": 10}
                          ])
    '''pipe = [{"$unwind": "$topics"},
            {"$group":
                {"_id": '$_id',
                 "averageScore": {"$avg": "$topics.score"},
                 "name": {'$first': '$name'}
                 }
             },
            {"$sort": {"averageScore": -1}}
            ]
    logs.aggregate(pipe)'''
    # ips = logs.count('ip')#.sort("ip", -1)
    # #.limit(10)#logs.sort("ip", -1).limit(10)
    # for document in logs:
    #    print(document.keys())
    get = nginx.count_documents({'method': 'GET'})
    post = nginx.count_documents({'method': 'POST'})
    put = nginx.count_documents({'method': 'PUT'})
    patch = nginx.count_documents({'method': 'PATCH'})
    delete = nginx.count_documents({'method': 'DELETE'})
    status = nginx.count_documents({'method': 'GET', 'path': '/status'})
    print("{} logs".format(logs))
    print("Methods:\n" +
          "\tmethod GET: {}\n".format(get) +
          "\tmethod POST: {}\n".format(post) +
          "\tmethod PUT: {}\n".format(put) +
          "\tmethod PATCH: {}\n".format(patch) +
          "\tmethod DELETE: {}".format(delete))
    print("{} status check".format(status))
    print("IPs:")
    for doc in ips:
        print("\t{}: {}".format(doc.get('_id'), doc.get('sum')))
