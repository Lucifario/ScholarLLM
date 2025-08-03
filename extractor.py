import fitz, pdfplumber, PIL.Image, io, re, uuid, datetime, json, base64, pytesseract, torch, numpy as np
from typing import List,Optional,Dict,Any,Tuple
from PIL import Image
from transformers import CLIPProcessor,CLIPModel

pytesseract.pytesseract.tesseract_cmd='/opt/homebrew/bin/tesseract'
CLIP_PROCESSOR=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIP_MODEL=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def generate_block_id(page_number:int,block_number:int,prefix:str="b")->str:
    """
    Generate a unique block ID based on page number and block number.
    """

    return f"{page_number+1}-{prefix}{block_number}"

def find_heading_level(font_size:Optional[float],font_name:Optional[str],size_thresholds:Dict[str,float])->int:
    """
    Determine the heading level based on font size and font name.
    """

    if font_size is None or font_name is None:
        return 0
    
    if "bold" in font_name.lower() or "black" in font_name.lower():
        if font_size>=size_thresholds.get("h1",16):return 1
        elif font_size>=size_thresholds.get("h2",14):return 2
        elif font_size>=size_thresholds.get("h3",12):return 3

    return 0

def extract_citations(text:str)->Optional[List[str]]:
    """
    Extract citations from the text using a regex pattern.
    """

    if not text:
        return None
    
    citation_patterns=[
        r'\[\d+(?:,\s*\d+)*\]',r'\[\d+-\d+\]',
        r'\((?:[A-Za-zÀ-ÖØ-öø-ÿ]+\s*et al\.?|\s*[A-Za-zÀ-ÖØ-öø-ÿ]+)(?:,\s*\d{4}[a-z]?)(?:;\s*(?:[A-Za-zÀ-ÖØ-öø-ÿ]+\s*et al\.?|\s*[A-Za-zÀ-ÖØ-öø-ÿ]+)(?:,\s*\d{4}[a-z]?))*\)',
        r'\([A-Za-zÀ-ÖØ-öø-ÿ]+,\s*[A-Za-zÀ-ÖØ-ööø-ÿ]+\s*and\s*[A-Za-zÀ-ÖØ-öø-ÿ]+,\s*\d{4}[a-z]?\)',
        r'\([A-Za-zÀ-ÖØ-öø-ÿ]+,\s*[A-Za-zÀ-ÖØ-öø-ÿ]+\s*,\s*\d{4}[a-z]?\)',
    ]

    matches=[]

    for pattern in citation_patterns:
        matches.extend(re.findall(pattern,text))

    return matches if matches else None

def classify_section(heading:Optional[str])->str:
    """
    Classify the section based on the heading.
    """

    if not heading:
        return "unknown"
    
    heading=heading.lower().strip()

    section_keywords={
        "abstract":["abstract","summary"],"introduction":["introduction","intro","background"],
        "methods":["method","methodology","materials","experimental","approach"],
        "results":["results","findings","outcomes","experiments"],
        "discussion":["discussion","analysis","interpretation"],
        "conclusion":["conclusion","conclusions","summary","final remarks"],
        "references":["references","bibliography","citations","works cited"],
        "appendix":["appendix","appendices","supplementary material"]
    }

    for section,keywords in section_keywords.items():
        if any(keyword in heading for keyword in keywords):return section

    return "other"

def caption_near_heading(blocks:List[Dict[str,Any]],target_block:Dict[str,Any])->Optional[str]:
    """
    Find a caption near the target,a table or an image block.
    """

    target_page=target_block['page_number']
    target_idx=-1

    for i,block in enumerate(blocks):
        if block.get('block_id')==target_block.get('block_id'):
            target_idx=i
            break
    if target_idx==-1:return None
    search_range=5
    caption_indicators=["figure","fig.","table","tbl."]

    for i in range(max(0,target_idx - search_range),target_idx):
        block=blocks[i]

        if (block["type"]=="text" and block["page_number"]==target_page):
            content_lower=block["content"].lower().strip()

            if any(indicator in content_lower for indicator in caption_indicators) and len(content_lower)<200:
                if block['bbox'][3]<target_block['bbox'][1]:return block["content"]

    for i in range(target_idx+1,min(len(blocks),target_idx+search_range+1)):
        block=blocks[i]

        if (block["type"]=="text" and block["page_number"]==target_page):
            content_lower=block["content"].lower().strip()

            if any(indicator in content_lower for indicator in caption_indicators) and len(content_lower)<200:
                if block['bbox'][1]>target_block['bbox'][3]:return block["content"]

    return None

def process_table_content(table_data:List[List[str]])->str:
    """
    Flattens and converts the tabular data to embed with text.
    """
    if not table_data:
        return ""
    markdown_output = ""
    if table_data[0]:
        header = [str(item) if item is not None else "" for item in table_data[0]]
        markdown_output += "| " + " | ".join(header) + " |\n"
        markdown_output += "|---" * len(header) + "|\n"

    for row in table_data[1:]:
        processed_row = [str(item) if item is not None else "" for item in row]
        markdown_output += "| " + " | ".join(processed_row) + " |\n"

    return markdown_output.strip()

def process_image_content(image_bytes:bytes,image_id:str)->Dict[str,Any]:
    """
    To process an image,extract OCR text and CLIP embeddings.
    """

    result={"ocr_text":None,"clip_embedding":None}

    try:
        image=Image.open(io.BytesIO(image_bytes))
        ocr_text=pytesseract.image_to_string(image)

        if ocr_text:
            result["ocr_text"]=ocr_text.strip()

        else:
            result["ocr_text"]=""

        inputs=CLIP_PROCESSOR(images=image,return_tensors="pt")

        with torch.no_grad():
            image_features=CLIP_MODEL.get_image_features(**inputs)

        result["clip_embedding"]=image_features.squeeze().tolist()

    except Exception as e:
        print(f"Error processing image {image_id}:{e}")

    return result

def extract_paper_content_metadata(doc:fitz.Document,min_blocks_to_scan:int=20)->Dict[str,Any]:
    """
    Extracts metadata from the first few pages of the PDF document.
    """

    metadata={
        "title":"Unknown Title",
        "authors":["Unknown Author"],
        "publication_year":0,
        "venue":"Unknown Venue",
        "abstract":""
    }

    doc_metadata=doc.metadata

    if doc_metadata:
        if doc_metadata.get("title"):metadata["title"]=doc_metadata["title"].strip()

        if doc_metadata.get("author"):
            authors_str=doc_metadata["author"].replace(";",",").replace(" and ",",").strip()
            metadata["authors"]=[a.strip() for a in authors_str.split(',') if a.strip()]
            metadata["authors"]=[a for a in metadata["authors"] if a.lower() not in ["et al","anon"]]

        if doc_metadata.get("creationDate"):
            try:
                year_match=re.search(r'D:(\d{4})',doc_metadata["creationDate"])
                if year_match:metadata["publication_year"]=int(year_match.group(1))
            except Exception:pass

    first_page_blocks_content=[]
    blocks_processed=0

    for page_num in range(min(doc.page_count,3)):
        page=doc[page_num]
        text_d=page.get_text("dict")
        text_b=text_d["blocks"]        

        for i,block in enumerate(text_b):
            if block["type"]!=0:continue

            content=" ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()

            if content:
                font_sizes=[span["size"] for line in block["lines"] for span in line["spans"]]
                font_names=[span["font"] for line in block["lines"] for span in line["spans"]]
                first_page_blocks_content.append({
                    "content":content,
                    "bbox":block["bbox"],
                    "font_size":sum(font_sizes)/len(font_sizes) if font_sizes else 0,
                    "font_name":font_names[0] if font_names else "Unknown"
                })
                blocks_processed+=1

                if blocks_processed>=min_blocks_to_scan:break

        if blocks_processed>=min_blocks_to_scan:break

    first_page_blocks_content.sort(key=lambda x:x['bbox'][1])

    for block in first_page_blocks_content:
        if block['font_size']>=20 and ("bold" in block['font_name'].lower() or "black" in block['font_name'].lower()):
            if len(block['content'].split())<20:
                metadata["title"]=block['content']
                break

    authors_found=False

    for i,block in enumerate(first_page_blocks_content):
        if authors_found:break
        if "Unknown Title" not in metadata["title"] and metadata["title"]==block['content']:

            for j in range(i + 1,min(len(first_page_blocks_content),i + 5)):
                candidate_block=first_page_blocks_content[j]

                if "@" in candidate_block['content'] or "university" in candidate_block['content'].lower() or "institute" in candidate_block['content'].lower():
                    potential_authors_text=first_page_blocks_content[j-1]['content'] if j-1 >=0 else candidate_block['content']
                    split_authors=re.split(r',| and ',potential_authors_text)
                    cleaned_authors=[a.strip() for a in split_authors if a.strip() and len(a.strip().split())>1 and not re.search(r'\d',a)]

                    if len(cleaned_authors)>0:
                        metadata["authors"]=cleaned_authors
                        authors_found=True; break
                    
                elif (candidate_block['font_size']<14 and candidate_block['font_size']>9 and len(candidate_block['content'].split())<50):
                    potential_authors_text=candidate_block['content']

                    if re.search(r'[A-Z][a-z]+\s[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s[A-Z][a-z]+)*',potential_authors_text):
                        split_authors=re.split(r',| and ',potential_authors_text)
                        cleaned_authors=[a.strip() for a in split_authors if a.strip() and len(a.strip().split())>1]

                        if len(cleaned_authors)>0:
                            metadata["authors"]=cleaned_authors
                            authors_found=True; break

    abstract_content_list=[]
    in_abstract_section=False
    
    for block in first_page_blocks_content:

        if block.get('content') and classify_section(block['content'])=='abstract':
            in_abstract_section=True; continue
        
        if in_abstract_section and block['content'] and not block['font_size']>12:
            if classify_section(block['content']) != 'unknown' and block['font_size']>=14:break
            abstract_content_list.append(block['content'])

        elif in_abstract_section and (not block['content'] or block['font_size']>14):break
            
    if abstract_content_list:metadata["abstract"]="\n".join(abstract_content_list).strip()

    year_match=re.search(r'\b(19|20)\d{2}\b'," ".join([b['content'] for b in first_page_blocks_content]))
    if year_match and metadata["publication_year"]==0:metadata["publication_year"]=int(year_match.group(0))

    return metadata

def parse_reference_entry(ref_text:str)->Dict[str,Any]:
    """
    Heuristically parses a single reference string into structured components.
    This is a simplified approach; true parsing is very complex.
    """

    parsed_ref={
        "raw_text":ref_text,
        "title":None,
        "authors":[],
        "year":None,
        "journal_venue":None,
        "volume_issue_pages":None,
        "doi":None
    }

    year_match=re.search(r'\(?(\d{4}[a-z]?)\)?\.?',ref_text)

    if year_match:
        parsed_ref["year"]=int(year_match.group(1)[:4])
        ref_text=re.sub(r'\(?(\d{4}[a-z]?)\)?\.?','',ref_text,1).strip()

    doi_match=re.search(r'(?:doi|DOI):\s*(\S+)',ref_text)

    if doi_match:
        parsed_ref["doi"]=doi_match.group(1).strip()
        ref_text=re.sub(r'(?:doi|DOI):\s*(\S+)','',ref_text,1).strip()

    parts=[p.strip() for p in re.split(r'\.\s*|\:\s*',ref_text) if p.strip()]

    if parts:
        potential_authors_str=parts[0]
        if re.search(r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]\.?)+)',potential_authors_str) or \
           re.search(r'([A-Z]\.\s*[A-Z][a-z]+(?:\s+and\s+[A-Z]\.\s*[A-Z][a-z]+)*)',potential_authors_str):
            authors_list=[a.strip() for a in re.split(r',| and ',potential_authors_str) if a.strip()]
            parsed_ref["authors"]=authors_list
            parts.pop(0)

    if parts:
        if 5<len(parts[0].split())<30:
            parsed_ref["title"]=parts[0]
            parts.pop(0)

    if parts:
        remaining_text=" ".join(parts).strip()
        journal_match=re.search(r'\b(?:Journal|Conf|Proc|Transactions|arXiv|Nature|Science)\b.*',remaining_text,re.IGNORECASE)
        if journal_match:
            parsed_ref["journal_venue"]=journal_match.group(0).strip()
            remaining_text=remaining_text.replace(journal_match.group(0),'',1).strip()

        volume_issue_pages_match=re.search(r'\b\d+\s*(?:\(\d+\))?:\s*\d+-\d+\b',remaining_text)

        if volume_issue_pages_match:
            parsed_ref["volume_issue_pages"]=volume_issue_pages_match.group(0).strip()
            remaining_text=remaining_text.replace(volume_issue_pages_match.group(0),'',1).strip()
    
    return parsed_ref

def extract_references_from_bibliography(all_blocks:List[Dict[str,Any]])->List[Dict[str,Any]]:
    """
    Extracts and parses all reference entries from the bibliography section(s).
    """
    references_section_blocks=[]
    
    for block in all_blocks:
        if block.get('toc_section')=='references' and block['type']=='text':
            references_section_blocks.append(block)

    if not references_section_blocks:
        print("No 'references' section found or no text blocks within it.")
        return []

    references_section_blocks.sort(key=lambda x:(x['page_number'],x['bbox'][1]))
    
    parsed_references=[]
    current_ref_lines=[]
    ref_counter=0

    for i,block in enumerate(references_section_blocks):
        for line in block['lines']:
            line_stripped=line.strip()
            if not line_stripped:continue

            is_new_ref_start=False
            if re.match(r'^\s*\[\d+\]\s*',line_stripped) or \
               re.match(r'^\s*\d+\.\s*',line_stripped) or \
               re.match(r'^\s*\d+\s+',line_stripped):

                if current_ref_lines and (line.index(line_stripped[0])<(references_section_blocks[i-1]['lines'][-1] if i>0 else "").index(references_section_blocks[i-1]['lines'][-1].strip()[0] if references_section_blocks[i-1]['lines'][-1].strip() else "") if references_section_blocks[i-1]['lines'] else 0):
                    is_new_ref_start=True

                elif not current_ref_lines:
                    is_new_ref_start=True

                elif i>0 and references_section_blocks[i-1]['page_number'] != block['page_number'] and re.match(r'^\s*\[?\d+\]?\s*',line_stripped):
                    is_new_ref_start=True

            if is_new_ref_start and current_ref_lines:
                ref_text=" ".join(current_ref_lines).strip()

                if ref_text:
                    parsed_references.append(parse_reference_entry(ref_text))

                current_ref_lines=[]
                ref_counter += 1
            
            current_ref_lines.append(line_stripped)

    if current_ref_lines:
        ref_text=" ".join(current_ref_lines).strip()

        if ref_text:
            parsed_references.append(parse_reference_entry(ref_text))

    return parsed_references


def extract_page_content(pdf_path:str,page_number:int,paper_id:str,timestamp:str,paper_details:Dict[str,Any])->List[Dict[str,Any]]:
    json_blocks=[]
    pdf=fitz.open(pdf_path)
    page=pdf[page_number]

    text_d=page.get_text("dict")
    text_b=text_d["blocks"]

    heading_size_thresholds={"h1":paper_details.get("title_font_size",16),"h2":14,"h3":12}

    for i,block in enumerate(text_b):
        if block["type"] != 0:continue
        lines=[]; content=""; font_sizes=[]; font_names=[]
        for line in block["lines"]:
            line_text="";
            for span in line["spans"]:
                line_text+=span["text"]
                font_sizes.append(span["size"])
                font_names.append(span["font"])
            lines.append(line_text)
            content+=line_text+"\n"

        content=content.strip()
        if not content:continue

        avg_font_size=sum(font_sizes)/len(font_sizes) if font_sizes else None
        primary_font=font_names[0] if font_names else None
        heading_level=find_heading_level(avg_font_size,primary_font,heading_size_thresholds)

        block_dict={
            "paper_id":paper_id,"page_number":page_number + 1,"block_id":generate_block_id(page_number,i),
            "type":"text","bbox":block["bbox"],"content":content,"lines":lines,
            "font_size":avg_font_size,"font_name":primary_font,"heading_level":heading_level,
            "timestamp":timestamp,"citations":extract_citations(content),
            "section":classify_section(content.split("\n")[0] if content else None),
            "paper_title":paper_details["title"],"paper_authors":paper_details["authors"],
            "paper_year":paper_details["publication_year"],"paper_venue":paper_details["venue"],
            "paper_abstract_summary":paper_details["abstract"][:500]
        }

        json_blocks.append(block_dict)

    image_list=page.get_images(full=True)
    for img_index,img in enumerate(image_list):
        xref=img[0]
        base_image=pdf.extract_image(xref)
        img_rects=page.get_image_rects(xref)
        bbox=list(img_rects[0]) if img_rects else [0,0,0,0]
        processed_img_data=process_image_content(base_image["image"],f"p{page_number+1}-img{img_index}")

        image_dict={
            "paper_id":paper_id,"page_number":page_number + 1,"block_id":generate_block_id(page_number,img_index,prefix="i"),
            "type":"image","bbox":bbox,
            "content":str(processed_img_data.get("ocr_text", "")),
            "image_id":f"p{page_number+1}-img{img_index}","bytes_b64":base64.b64encode(base_image["image"]).decode('utf-8'),
            "clip_embedding":processed_img_data.get("clip_embedding"),"timestamp":timestamp,
            "paper_title":paper_details["title"],"paper_authors":paper_details["authors"],
            "paper_year":paper_details["publication_year"],"paper_venue":paper_details["venue"],
            "paper_abstract_summary":paper_details["abstract"][:500]
        }
        json_blocks.append(image_dict)

    pdf.close()

    pdf=pdfplumber.open(pdf_path)
    if page_number<len(pdf.pages):
        plumber_page=pdf.pages[page_number]
        page_tables=plumber_page.extract_tables()

        if page_tables:
            for table_index,table in enumerate(page_tables):
                flattened_table_text=process_table_content(table)
                
                table_dict={
                    "paper_id":paper_id,"page_number":page_number + 1,"block_id":generate_block_id(page_number,table_index,prefix="t"),
                    "type":"table","bbox":plumber_page.bbox,
                    "content_raw":table,
                    "content_flattened":str(flattened_table_text),
                    "table_id":f"p{page_number+1}-tbl{table_index}","timestamp":timestamp,
                    "paper_title":paper_details["title"],"paper_authors":paper_details["authors"],
                    "paper_year":paper_details["publication_year"],"paper_venue":paper_details["venue"],
                    "paper_abstract_summary":paper_details["abstract"][:500]
                }
                json_blocks.append(table_dict)

    pdf.close()
    return json_blocks

def assign_nearest_heading(blocks:List[Dict[str,Any]])->List[Dict[str,Any]]:
    blocks.sort(key=lambda x:(x['page_number'],x['bbox'][1]))
    last_heading=None
    current_section="unknown"
    for block in blocks:
        if block["type"]=="text" and block.get("heading_level",0)>0:
            last_heading=block["content"]
            current_section=classify_section(last_heading)
        block["nearest_heading"]=last_heading
        block["toc_section"]=current_section
    return blocks

def assign_captions(blocks: List[Dict]) -> None:
    for block in blocks:
        if block["type"] in ["image", "table"]:
            caption = caption_near_heading(blocks, block)
            block["caption"] = str(caption or "").strip()

def page_json_output(page_blocks:List[Dict[str,Any]],page_number:int)->Dict[str,Any]:
    print(f"\n{'='*60}")
    print(f"PAGE {page_number} - COMPLETE JSON OUTPUT")
    print(f"{'='*60}")
    serializable_blocks=[]
    for block in page_blocks:
        block_copy=block.copy()
        if "bytes" in block_copy:del block_copy["bytes"]
        if "bytes_b64" in block_copy and block_copy["bytes_b64"]:
            encoded=block_copy["bytes_b64"]
            block_copy["bytes_b64"]=f"{encoded[:50]}... (truncated,full length:{len(encoded)})"
        if "clip_embedding" in block_copy and block_copy["clip_embedding"] is not None:
            block_copy["clip_embedding"]=f"embedding_length_{len(block_copy['clip_embedding'])}"
        serializable_blocks.append(block_copy)
    json_output=json.dumps(serializable_blocks,indent=2,ensure_ascii=False)
    print(json_output)
    print(f"{'='*60}")
    print(f"PAGE {page_number} SUMMARY:{len(page_blocks)} blocks extracted")
    print(f"{'='*60}\n")


def extract_structured_pdf(pdf_path:str,print_per_page:bool=False,print_json_per_page:bool=False)->Dict[str,Any]:
    """
    Extracts structured content from PDF,including paper-level metadata and parsed bibliography.
    Returns a dictionary containing all blocks and parsed references.
    """

    paper_id=uuid.uuid4().hex
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    all_blocks=[]

    with fitz.open(pdf_path) as doc:
        total_pages=len(doc)
        paper_details=extract_paper_content_metadata(doc)
        print(f"Extracted Paper Details:Title='{paper_details['title']}',Authors={paper_details['authors']},Year={paper_details['publication_year']},Venue='{paper_details['venue']}'")
    
    print(f"Processing {total_pages} pages...")
    
    for page_num in range(total_pages):
        print(f"\nProcessing page {page_num + 1}/{total_pages}")
        page_blocks=extract_page_content(pdf_path,page_num,paper_id,timestamp,paper_details)
        all_blocks.extend(page_blocks)
        
        if print_json_per_page:
            page_json_output(page_blocks,page_num + 1)
        elif print_per_page:
            print(f"PAGE {page_num + 1} SUMMARY:{len(page_blocks)} blocks extracted")
    
    print("\nPost-processing:assigning nearest headings and captions...")
    all_blocks=assign_nearest_heading(all_blocks)
    assign_captions(all_blocks)
    
    print("\nExtracting and parsing bibliography references...")
    parsed_references=extract_references_from_bibliography(all_blocks)
    print(f"Found and parsed {len(parsed_references)} references.")

    print(f"Extraction complete. Found {len(all_blocks)} blocks.")

    return {
        "paper_id":paper_id,
        "timestamp":timestamp,
        "paper_details":paper_details,# Top-level paper metadata
        "blocks":all_blocks,
        "parsed_references":parsed_references # Structured bibliography
    }


def save_extracted_data_to_json(data:Dict,output_path:str)->None:
    """
    Save the complete extracted data dictionary to a JSON file.
    Handles bytes serialization.
    """

    serializable_data=json.loads(json.dumps(data)) 

    for block in serializable_data.get('blocks',[]):
        if "bytes" in block:
            del block["bytes"]
        if "bytes_b64" in block and block["bytes_b64"]:
            pass 
        if "clip_embedding" in block and block["clip_embedding"] is not None:
            pass 
        for k,v in block.items():
            if isinstance(v,np.ndarray):
                block[k]=v.tolist()

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(serializable_data,f,indent=2,ensure_ascii=False)

"""
if __name__=="__main__":
    pdf_path="test.pdf"
    output_json_path="extracted_paper_data.json"

    print("=== STARTING COMPLETE PDF EXTRACTOR ===")
    extracted_data=extract_structured_pdf(pdf_path,print_json_per_page=False)
    
    save_extracted_data_to_json(extracted_data,output_json_path)
    print(f"\nExtraction complete. All structured data saved to {output_json_path}")

    print("\n--- Sample Parsed References ---")
    for i,ref in enumerate(extracted_data['parsed_references'][:3]):
        print(f"Reference {i+1}:")
        print(f"  Title:{ref.get('title')}")
        print(f"  Authors:{','.join(ref.get('authors',[]))}")
        print(f"  Year:{ref.get('year')}")
        print(f"  DOI:{ref.get('doi','N/A')}")
        print("-" * 20)

    print("\nExtractor is now producing rich,structured output ready for graphing or RAG.")
"""