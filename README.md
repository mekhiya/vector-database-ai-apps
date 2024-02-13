# Vector Database AI Apps
AI Apps using Vector Database
(Pinecone)

Vector databses is eseential part of stack for developing LLM base applications. 
**RAG** - (retrieval augmented generation), retrieves the relevant data and use it as augmented context for the LLM application.

VECTOR DBs can also do:
- Text similarity search
- RAGs
- Image similarity search
- anamoly detection
- recommendation system

Vector dbs good for sparse & dense vectors

Repo consists of below 6 apps using Vector DBs in various ways: 
- 1) Basic semantic search for text documents
- 2) RAG
- 3) Recommendation system
- 4) Hybrid Search app for product Recommendation (uses dense vector for image & sparse for text)
- 5) Child Parent similarity app
- 6) Anamoly dtection based on database of server logs
 

## 1) SEMANTIC SEARCH
search using meaning of content being search, whereas lexical search which looks for literal or pattern matching strings. 

 ![SEMANTIC SEARCH](https://private-user-images.githubusercontent.com/8952786/304038683-cccb18b7-083c-4596-8ddf-e466b3f5f32a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc3Mzc3ODIsIm5iZiI6MTcwNzczNzQ4MiwicGF0aCI6Ii84OTUyNzg2LzMwNDAzODY4My1jY2NiMThiNy0wODNjLTQ1OTYtOGRkZi1lNDY2YjNmNWYzMmEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDIxMiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDAyMTJUMTEzMTIyWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MzE5YmRlMjJkY2EyNjM5NWQ2ZjJlZWE1Y2VlOWU5NTZlYjA5NjMyODNjNzliZTQzODQxYTkxZDc5YWI1ZGQyOCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.LaI4liEeF3nNiacM0HwMXdVsKYFfZVBWrXcy3brig5c)

 We will use Sentence Trasnformer model file for embedding.
 ## FROM [sbert.net](https://www.sbert.net/)
- SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings.(initial work - paper Sentence-BERT)
- framework to compute sentence / text embeddings for more than 100 languages. 
- embeddings can be compared e.g. with cosine-similarity to find sentences with a similar meaning.
- useful for semantic textual similarity, semantic search, or paraphrase mining.
- framework based on PyTorch and Transformers
- offers a large collection of pre-trained models tuned for various tasks. 
- easy to fine-tune your own models 

