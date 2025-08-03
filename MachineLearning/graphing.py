import networkx as nx
from neo4j import GraphDatabase
import re, spacy, hashlib
from collections import Counter

nlp_graph=spacy.load("en_core_web_sm")

def get_neo4j_match_key_value(nid_nx,nl_nx):

    if nl_nx == "Paper":

        if nid_nx.startswith("DOI::"): return "doi",nid_nx.replace("DOI::","")

        elif nid_nx.startswith("REF::") or nid_nx.startswith("RAWREF::"): return "paper_id",nid_nx

        else: return "paper_id",nid_nx

    elif nl_nx=="Author": return "name",nid_nx

    elif nl_nx=="Method": return "name",nid_nx

    elif nl_nx=="Result": return "id",nid_nx

    elif nl_nx=="Concept": return "name",nid_nx

    elif nl_nx=="Block": return "block_id",nid_nx

    return "id",nid_nx

def load_nx_graph_to_neo4j_tx(tx,graph_nx: nx.DiGraph):

    for nid,original_attrs in graph_nx.nodes(data=True):

        l=original_attrs.get("type","Unknown")
        props_for_neo4j={k:v for k,v in original_attrs.items() if k!="type"}
        
        if l=="Paper":

            if "doi" in props_for_neo4j and props_for_neo4j["doi"]:
                tx.run(
                    "MERGE (n:Paper {doi:$doi_prop}) "
                    "ON CREATE SET n.paper_id=$id,n += $props "
                    "ON MATCH SET n += $props",
                    id=nid,props=props_for_neo4j,doi_prop=props_for_neo4j["doi"]
                )

            else:
                tx.run(
                    "MERGE (n:Paper {paper_id:$id}) "
                    "SET n += $props",
                    id=nid,props=props_for_neo4j
                )

        elif l=="Author":
            tx.run(
                "MERGE (n:Author {name:$id})",
                id=nid
                )
            
        elif l=="Method":
            tx.run(
                "MERGE (n:Method {name:$id}) SET n += $props",
                id=nid,props=props_for_neo4j
            )

        elif l=="Result":
            tx.run(
                "MERGE (n:Result {id:$id}) SET n += $props",
                id=nid,props=props_for_neo4j
            )

        elif l=="Concept":
            tx.run(
                "MERGE (n:Concept {name:$id}) SET n += $props",
                id=nid,props=props_for_neo4j
            )

        elif l=="Block":
            tx.run(
                "MERGE (n:Block {block_id:$id}) SET n += $props",
                id=nid,props=props_for_neo4j
            )

        else:
            tx.run(
                f"MERGE (n:{l} {{id:$id}}) SET n += $props",
                id=nid,props=props_for_neo4j
            )

    for u,v,attributes in graph_nx.edges(data=True):

        edge_type=attributes.get("type")
        edge_data={k:v for k,v in attributes.items() if k != "type"}

        u_node_type=graph_nx.nodes[u].get("type","Unknown")
        v_node_type=graph_nx.nodes[v].get("type","Unknown")

        u_key,u_val=get_neo4j_match_key_value(u,u_node_type)
        v_key,v_val=get_neo4j_match_key_value(v,v_node_type)

        query=f"""
        MATCH (u:{u_node_type}),(v:{v_node_type})
        WHERE u.{u_key}=$u_val AND v.{v_key}=$v_val
        MERGE (u)-[r:{edge_type}]->(v)
        SET r += $props
        """

        tx.run(query,u_val=u_val,v_val=v_val,props=edge_data)

def build_and_load_paper_graph(extracted_data: dict,neo4j_driver_instance: GraphDatabase.driver):
    """
    Builds a NetworkX graph from extracted paper data and loads it into Neo4j.
    """

    paper_id=extracted_data.get("paper_id","unknown_paper")
    paper_metadata=extracted_data.get("paper_details",{})
    blocks=extracted_data.get("blocks",[])
    parsed_references=extracted_data.get("parsed_references",[])

    print(f"Building NetworkX graph for Paper ID: {paper_id}")

    graph=nx.DiGraph()

    graph.add_node(paper_id,type="Paper",title=paper_metadata.get("title","N/A"),year=paper_metadata.get("publication_year","N/A"),venue=paper_metadata.get("venue","N/A"),abstract=paper_metadata.get("abstract","N/A"))

    for author in paper_metadata.get("authors",[]):
        graph.add_node(author,type="Author",name=author)
        graph.add_edge(paper_id,author,type="AUTHORED_BY")

    for blk in blocks:

        content_for_hash = str(blk.get('content', '')).strip()
        block_id = blk.get("block_id", f"block_{hashlib.sha256(content_for_hash.encode('utf-8')).hexdigest()}")
        block_props={"section":blk.get("section","unknown"),"block_type":blk.get("type","unknown"),"page_number":blk.get("page_number",-1),"block_index":blk.get("bbox",-1),"content_snippet":blk.get("content","")[:200] + "..." if len(blk.get("content","")) > 200 else blk.get("content","")}
        graph.add_node(block_id,type="Block",**block_props)
        graph.add_edge(paper_id,block_id,type="PAPER_HAS_BLOCK")

    for ref in parsed_references:

        cited_doi=ref.get("doi")
        cited_title=ref.get("title")
        cited_authors=ref.get("authors",[])
        cited_year=ref.get("year")

        if cited_doi:
            cited_node_id=f"DOI::{cited_doi}"

        elif cited_title and cited_year:
            unique_string=f"{cited_title}_{cited_year}_{'_'.join(cited_authors)}".encode('utf-8')
            cited_node_id=f"REF::{hashlib.sha256(unique_string).hexdigest()}"

        else:
            unique_string=ref.get('raw_text',f"unknown_reference_{ref.get('index','unknown')}").encode('utf-8')
            cited_node_id=f"RAWREF::{hashlib.sha256(unique_string).hexdigest()}"

        cited_paper_props={"title":cited_title,"year":cited_year,"doi":cited_doi,"authors":cited_authors,"raw_reference_text":ref.get("raw_text")}
        graph.add_node(cited_node_id,type="Paper",**{k:v for k,v in cited_paper_props.items() if v is not None})
        graph.add_edge(paper_id,cited_node_id,type="CITES")

    all_text_content_for_nlp=[]

    for blk in blocks:
        block_id_for_link = blk.get("block_id", f"block_{hashlib.sha256(str(blk.get('content','')).encode('utf-8')).hexdigest()}")

        if blk.get("type") == "text":
            text_content = str(blk.get("content", "")).strip()
            all_text_content_for_nlp.append(text_content)

            if blk.get("section") == "methods":
                methods = re.findall(r"(?:we (?:use|propose|apply) )([A-Z][A-Za-z0-9\-\s]{3,})", text_content, flags=re.IGNORECASE)

                for m_name in set(methods):
                    m_name_cleaned=m_name.strip()

                    if m_name_cleaned and 1 < len(m_name_cleaned.split()) <= 5:
                        graph.add_node(m_name_cleaned,type="Method",name=m_name_cleaned)
                        graph.add_edge(block_id_for_link,m_name_cleaned,type="USES_METHOD",page_number=blk['page_number'])

            if blk.get("section") == "results":
                nums = re.findall(r"(\d+(?:\.\d+)?(?:%\s*| percent|x)?)\b", text_content, flags=re.IGNORECASE)

                for val in set(nums):
                    result_node_id=f"result_{blk['block_id']}_{val}"
                    graph.add_node(result_node_id,type="Result",value=val,block_id=blk['block_id'],page_number=blk['page_number'])
                    graph.add_edge(block_id_for_link,result_node_id,type="REPORTS",page_number=blk['page_number'])

    all_text_combined=" ".join(all_text_content_for_nlp)

    if all_text_combined:
        
        doc_nlp=nlp_graph(all_text_combined)
        phrases=Counter(chunk.text.lower() for chunk in doc_nlp.noun_chunks
                          if len(chunk.text.split()) > 1 and len(chunk.text) > 3 and not chunk.text.replace('.','').strip().isdigit())
        
        for phrase,freq in phrases.most_common(20):
            graph.add_node(phrase,type="Concept",name=phrase,frequency=freq)
            graph.add_edge(paper_id,phrase,type="MENTIONS",frequency=freq)

    with neo4j_driver_instance.session() as session:
        session.write_transaction(load_nx_graph_to_neo4j_tx,graph)
    print(f"Graph for Paper ID {paper_id} pushed to Neo4j.")