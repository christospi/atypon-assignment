"""
Configuration file for training setup
"""

props = {
    "__model_options__": ["logreg", "bilstm"],
    "__embeddings_options__": ["glove.6B.50d.txt", "glove.6B.100d.txt"],
    "dataset": "data/dataset/assignment_dataset.json",
    "embeddings": "data/embeddings/glove.6B.100d.txt",
    "model": "bilstm",
    "embeddings_shape": 100,
    "embeddings_max_length": 200
}
