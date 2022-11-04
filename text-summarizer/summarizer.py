from transformers import pipeline

summarizer = pipeline("summarization")

text = """
The S&P 500 and the Dow Jones Industrial Average both closed at record highs on Wednesday, as investors cheered a strong earnings season and a rebound in the U.S. economy.
"""
print(summarizer(text, max_length=130, min_length=30, do_sample=False))