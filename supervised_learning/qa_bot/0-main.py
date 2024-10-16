#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('C:/Users/CAMPUSNA/Desktop/holbertonschool-machine_learning/supervised_learning/qa_bot/ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
