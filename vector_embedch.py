import json
import hashlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def embed_and_load_chroma_data(extracted_data: dict,chromadb_instance: Chroma):
    """
    Extracts relevant content from extracted_data and embeds it into ChromaDB.
    """

    blocks=extracted_data.get("blocks",[])
    paper_id=extracted_data.get("paper_id","unknown_paper")
    paper_details=extracted_data.get("paper_details",{})

    texts,metadatas,ids=[],[],[]

    for blk in blocks:
        btype=blk.get("type","unknown")
        content=""

        if btype in ("text","image"):
            content=str(blk.get("content", "")).strip()

        elif btype == "table":
            content=str(blk.get("content_flattened","")).strip()

        else:
            continue

        if not content:
            continue

        texts.append(content)
        metadata={
            "paper_id": paper_id,
            "paper_title": paper_details.get("title","N/A"),
            "block_id": blk.get("block_id","N/A"),
            "section": blk.get("section","unknown"),
            "block_type": btype,
            "page_number": blk.get("page_number",-1),
            "content_snippet": content[:200] + "..." if len(content) > 200 else content,
        }

        if blk.get("caption"):
            metadata["caption"]=str(blk["caption"].strip())

        metadatas.append(metadata)
        ids.append(blk.get("block_id", f"block_{hashlib.sha256(content.encode()).hexdigest()}"))

    if texts:
        chromadb_instance.add_texts(texts,metadatas=metadatas,ids=ids)
        print(f"Added {len(texts)} documents from Paper ID {paper_id} to ChromaDB.")

    else:
        print(f"No textual content found for Paper ID {paper_id} to embed.")