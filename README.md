# Calibration Test

This repo is a small, self-contained experiment for measuring model calibration on a test made by a human. 

prompt.txt: the test.

correct_answers.txt: the answers to the test.

answers1.jsonl: ChatGPT 5.2 answers in 50 runs to the test collected through the OpenAI API.

analyze_calibration.py: analysis script. Reads model responses + confidence, compares against ground truth, and reports calibration metrics.




