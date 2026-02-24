\# AI RAG-Based Intelligent Assistant



\## Overview

This project implements a basic Retrieval-Augmented Generation (RAG) style semantic search system using:



\- SentenceTransformers

\- FAISS (Vector Similarity Search)

\- Python



\## How It Works



1\. Load custom knowledge document

2\. Split document into chunks

3\. Convert chunks into embeddings

4\. Store embeddings in FAISS vector index

5\. Convert user query into embedding

6\. Retrieve most semantically similar chunk



\## Tech Stack

\- Python

\- SentenceTransformers (all-MiniLM-L6-v2)

\- FAISS

\- NumPy



\## How to Run



pip install sentence-transformers faiss-cpu numpy  

python main.py



\## Learning Outcome

This project demonstrates semantic retrieval using vector embeddings and forms the retrieval component of a RAG pipeline.

