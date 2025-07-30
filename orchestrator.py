import os
from neo4j import GraphDatabase
from langchain_community.vectorstores import Chroma
from extractor import extract_structured_pdf
from vector_embedch import embed_and_load_chroma_data, embedding_function
from graphing import build_and_load_paper_graph

NEO4J_URI=os.getenv('NEO4J_AURA_URL')
NEO4J_USERNAME=os.getenv('NEO4J_AURA_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_AURA_PASSWORD')

pdf_dir="pdfs"
CHROMA_PERSIST_DIRECTORY="chroma_db"
CHROMA_COLLECTION_NAME="embed_storage"

neo4j_driver_instance=GraphDatabase.driver(NEO4J_URI,auth=(NEO4J_USERNAME,NEO4J_PASSWORD))
chromadb_instance=Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    embedding_function=embedding_function
)

def pipeline(pdfs: str):
    failed_count=0
    proc_count=0

    for pdf_file in os.listdir(pdfs):
        if pdf_file.endswith(".pdf"):
            pdf_path=os.path.join(pdfs,pdf_file)
            print(f"Processing {pdf_path}...")

            try:
                extracted_data=extract_structured_pdf(pdf_path)
                build_and_load_paper_graph(extracted_data,neo4j_driver_instance)
                embed_and_load_chroma_data(extracted_data,chromadb_instance)
                proc_count += 1
            except Exception as e:
                print(f"Failed to process {pdf_path}: {e}")
                failed_count += 1

    chromadb_instance.persist()
    if neo4j_driver_instance:
        neo4j_driver_instance.close()
    print(f"Processed {proc_count} PDFs successfully, {failed_count} failed.")
    print(f"Total documents in ChromaDB collection: {chromadb_instance._collection.count()} (after all processing)")

if __name__ == "__main__":
    pipeline(pdf_dir)