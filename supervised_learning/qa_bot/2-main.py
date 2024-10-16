#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('C:/Users/CAMPUSNA/Desktop/holbertonschool-machine_learning/supervised_learning/qa_bot/ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
