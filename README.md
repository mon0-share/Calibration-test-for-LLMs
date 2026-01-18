# Calibration Test

This repo is a small, self-contained experiment for measuring model calibration on a test made by a human. 

prompt.txt
The test.

correct_answers.txt
The answers to the test.

answers1.jsonl
ChatGPT 5.2 answers in 50 runs collected through the OpenAI API.

analyze_calibration.py
Analysis script. Reads model responses + confidence, compares against ground truth, and reports calibration metrics.




