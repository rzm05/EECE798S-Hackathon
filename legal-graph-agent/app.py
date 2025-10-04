# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from knowledge_graph_maker import Ontology, Document
import torch, json, re, datetime, io, sys
import networkx as nx
import gradio as gr
from PIL import Image
from io import BytesIO
import os, uuid, textwrap
from pypdf import PdfReader
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Load model
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print(f"Hugging Face token loaded: {hf_token[:8]}...")
else:
    print("HF token not found. Set HF_TOKEN env var to pull gated models if needed.")

model_name = os.getenv("MODEL_NAME", "tiiuae/Falcon3-1B-Instruct")


dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map="auto",
)


if getattr(model.generation_config, "pad_token_id", None) is None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# Deterministic generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=dtype,
    device_map="auto",
    max_new_tokens=512,
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.05
)

# Prompt Template
class HuggingFaceClient:
    def __init__(self, generator):
        self.generator = generator

    def generate(self, user_message, system_message=""):
        prompt = f"""{system_message}

Return ONLY a JSON array. Use DOUBLE-QUOTED keys. Emit AT LEAST 4 edges per chunk.
Use labels ONLY from: ["Party","Obligation","Right","Clause","Date","Amount","Service","Condition","ConfidentialInfo","Miscellaneous"].
Use relationships ONLY from: ["Party has Obligation","Party holds Right","Clause references Clause","Condition triggers Obligation/Right","Party responsible for Service","Party pays Amount","Party protected by ConfidentialInfo"].
Prefer concrete names (e.g., company names, dates, amounts) over generic placeholders.

Valid edge schema:
[
  {{
    "node_1": {{"label":"Party","name":"TELCOSTAR PTE, LTD."}},
    "node_2": {{"label":"Party","name":"Ability Computer & Software Industries Ltd."}},
    "relationship": "Party has Obligation",
    "description": "Each party has duties under the Services Agreement."
  }},
  {{
    "node_1": {{"label":"Clause","name":"Services Agreement"}},
    "node_2": {{"label":"Date","name":"Effective Date: November 1, 2019"}},
    "relationship": "Clause references Clause"
  }},
  {{
    "node_1": {{"label":"Party","name":"Recipient"}},
    "node_2": {{"label":"Amount","name":"Actual cost + 10% service fee"}},
    "relationship": "Party pays Amount",
    "description": "Monthly invoices payable within 15 days."
  }}
]

Guidelines:
- If the text mentions “Effective Date”, output it as a Date node.
- If the text mentions payment terms or setoff, map to “Party pays Amount”.
- If the text mentions services delivered, map to “Party responsible for Service”.
- If the text mentions confidentiality, map to “Party protected by ConfidentialInfo”.
- If the text mentions termination conditions or acceptance/rejection windows, use “Condition triggers Obligation/Right”.
- Use “Clause references Clause” when a section references another section or legal concept.

User: {user_message}
Assistant:"""
        out = self.generator(
            prompt,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_full_text=False
        )
        text = out[0]["generated_text"]

        return text.strip()

llm = HuggingFaceClient(generator)


# Oncology (Structure of the graph)
ontology = Ontology(
    labels=[
        {"Party": "A legal entity involved in the agreement (individual or organization)."},
        {"Obligation": "A duty or responsibility imposed on a Party under this Agreement."},
        {"Right": "A legal right or entitlement held by a Party."},
        {"Clause": "A numbered or named section of the Agreement."},
        {"Date": "Dates referenced in the Agreement."},
        {"Amount": "Monetary amounts, fees, or penalties."},
        {"Service": "Specific service or deliverable provided under the Agreement."},
        {"Condition": "Conditions that trigger obligations, rights, or termination."},
        {"ConfidentialInfo": "Information marked as confidential in the Agreement."},
        {"Miscellaneous": "Other important concepts not covered by above categories."}
    ],
    relationships=[
        "Party has Obligation",
        "Party holds Right",
        "Clause references Clause",
        "Condition triggers Obligation/Right",
        "Party responsible for Service",
        "Party pays Amount",
        "Party protected by ConfidentialInfo"
    ]
)

def _labels_from_ontology(ont: Ontology):
    allowed = set()
    for item in ont.labels:
        if isinstance(item, str):
            allowed.add(item)
        elif isinstance(item, dict):
            allowed.update(item.keys())
    return allowed

ALLOWED_LABELS = _labels_from_ontology(ontology)
ALLOWED_RELATIONSHIPS = {
    "Party has Obligation",
    "Party holds Right",
    "Clause references Clause",
    "Condition triggers Obligation/Right",
    "Party responsible for Service",
    "Party pays Amount",
    "Party protected by ConfidentialInfo",
}

# Common alias mapping : legal ontology labels (used to process the output to have consistent naming of the labels)
ALIAS_TO_LABEL = {

    "provider": "Party", "recipient": "Party", "subcontractor": "Party",
    "party": "Party", "parties": "Party", "company": "Party",
    "organization": "Party", "organisation": "Party",

    "agreement": "Clause", "contract": "Clause", "services agreement": "Clause",
    "exhibit": "Clause", "section": "Clause", "clause": "Clause",

    "effective date": "Date", "date": "Date",
    "fee": "Amount", "fees": "Amount", "amount": "Amount", "payment": "Amount",
    "service": "Service",
    "confidential information": "ConfidentialInfo", "confidentialinfo": "ConfidentialInfo",
    "condition": "Condition", "obligation": "Obligation", "right": "Right",
}

# Json Parser
def _quote_unquoted_keys(s: str) -> str:
    return re.sub(r'(?P<sep>[{,\s])(?P<key>[A-Za-z_]\w*)\s*:', r'\g<sep>"\g<key>":', s)

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', s)

def _strip_code_fences(s: str) -> str:
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t

def _force_array(s: str) -> str:
    t = s.strip()
    if t.startswith('{') and t.endswith('}'):
        return f'[{t}]'
    return t

def extract_json_edges(output_str):
    cleaned = output_str.replace('<|assistant|>', '').strip()
    cleaned = _strip_code_fences(cleaned)
    cleaned = _quote_unquoted_keys(cleaned)
    cleaned = _strip_trailing_commas(cleaned)
    cleaned = _force_array(cleaned)


    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    objs = re.findall(r'\{.*?\}(?=\s*(?:\{|\]|$))', cleaned, flags=re.DOTALL)
    edges = []
    for obj in objs:
        try:
            edges.append(json.loads(_strip_trailing_commas(_quote_unquoted_keys(obj))))
        except json.JSONDecodeError:
            continue
    return edges

def _canonicalize_label(raw_label: str) -> str:
    if not raw_label:
        return "Miscellaneous"
    key = raw_label.strip().lower()
    if key in ALIAS_TO_LABEL:
        return ALIAS_TO_LABEL[key]
    if raw_label in ALLOWED_LABELS:
        return raw_label
    for lab in ALLOWED_LABELS:
        if lab.lower() == key:
            return lab
    return "Miscellaneous"

GENERIC_NAMES = {
    "provider","recipient","subcontractor","party","parties",
    "services agreement","agreement","clause","section","exhibit"
}

def _is_generic_name(name: str) -> bool:
    return (name or "").strip().lower() in GENERIC_NAMES

def _map_relationship(rel: str) -> str:
    """Coerce common phrasing into allowed relationships."""
    r = (rel or "").strip()
    if r in ALLOWED_RELATIONSHIPS:
        return r
    low = r.lower()
    if any(k in low for k in ["pay","invoice","fee","reimburse","amount"]):
        return "Party pays Amount"
    if any(k in low for k in ["obligation","shall","must","responsible","provide"]):
        return "Party has Obligation"
    if ("right" in low) or ("entitled" in low) or ("may" in low and "terminate" in low):
        return "Party holds Right"
    if any(k in low for k in ["confidential","confidentiality","non-disclosure"]):
        return "Party protected by ConfidentialInfo"
    if any(k in low for k in ["service","perform","provide","deliver"]):
        return "Party responsible for Service"
    if any(k in low for k in ["condition","force majeure","upon","if","accept","reject","notice"]):
        return "Condition triggers Obligation/Right"
    if any(k in low for k in ["clause","section","exhibit","agreement","references"]):
        return "Clause references Clause"
    return "Party has Obligation"

def sanitize_edges(edges_raw):
    """Validate edges and coerce labels/relationships to the ontology."""
    sanitized, seen = [], set()
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        n1, n2 = e.get("node_1"), e.get("node_2")
        rel = e.get("relationship") or e.get("relation") or e.get("edge")
        if not (isinstance(n1, dict) and isinstance(n2, dict) and isinstance(rel, str) and rel.strip()):
            continue

        l1 = _canonicalize_label(n1.get("label"))
        l2 = _canonicalize_label(n2.get("label"))
        name1 = (n1.get("name") or "").strip()
        name2 = (n2.get("name") or "").strip()
        if not name1 or not name2:
            continue


        if _is_generic_name(name1) and _is_generic_name(name2):
            if not (l1 in {"Clause","Date","Amount","Service"} or l2 in {"Clause","Date","Amount","Service"}):
                continue

        mapped_rel = _map_relationship(rel)
        edge = {
            "node_1": {"label": l1, "name": name1},
            "node_2": {"label": l2, "name": name2},
            "relationship": mapped_rel,
        }
        desc = e.get("description")
        if isinstance(desc, str) and desc.strip():
            edge["description"] = desc.strip()

        key = (edge["node_1"]["label"], edge["node_1"]["name"],
               edge["node_2"]["label"], edge["node_2"]["name"], edge["relationship"])
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(edge)
    return sanitized

def generate_edges(text, min_edges=3, max_retries=2):
    """LLM-only edge extraction for a text chunk."""
    SYS_PROMPT = (
        "Extract a knowledge graph from the text. "
        "Return ONLY a JSON array of edges. Emit AT LEAST 4 edges. "
        "Each edge MUST have keys: "
        "\"node_1\" (obj with \"label\",\"name\"), "
        "\"node_2\" (obj with \"label\",\"name\"), "
        "\"relationship\" (string), and optional \"description\". "
        f"Use ONLY these labels: {sorted(list(ALLOWED_LABELS))}. "
        f"Use ONLY these relationships (closest fit if needed): {sorted(list(ALLOWED_RELATIONSHIPS))}. "
        "Prefer concrete entity names (company names, dates, amounts)."
    )
    last_raw = ""
    for _ in range(max_retries + 1):
        last_raw = llm.generate(user_message=text, system_message=SYS_PROMPT)
        edges_raw = extract_json_edges(last_raw)
        edges = sanitize_edges(edges_raw)
        if len(edges) >= min_edges:
            return edges
    return sanitize_edges(extract_json_edges(last_raw))


# Split the text into chunks
def chunk_text(s, chunk_size=1400, overlap=200):
    chunks = []
    start, n = 0, len(s)
    while start < n:
        end = min(start + chunk_size, n)

        segment = s[start:end]
        dot = segment.rfind(".")
        if dot != -1 and end != n and dot > chunk_size * 0.6:
            end = start + dot + 1
            segment = s[start:end]
        chunks.append(segment)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_graph_from_text(text: str):
    """Run your pipeline on raw text and return (edges_list, networkx_graph)."""
    # Chunk
    chunks_local = chunk_text(text, chunk_size=1400, overlap=220)

    # Per-chunk edges
    all_edges_local = []
    for ch in chunks_local:
        all_edges_local.extend(generate_edges(ch, min_edges=2, max_retries=2))

    # Fallback on full text if needed
    if len(all_edges_local) < 8:
        full_edges = generate_edges(text, min_edges=8, max_retries=2)
        def key_of(e):
            return (e['node_1']['label'], e['node_1']['name'],
                    e['node_2']['label'], e['node_2']['name'], e['relationship'])
        seen = {key_of(e) for e in all_edges_local}
        for e in full_edges:
            if key_of(e) not in seen:
                all_edges_local.append(e); seen.add(key_of(e))

    # Graph
    G_local = nx.DiGraph()
    for e in all_edges_local:
        n1 = f"{e['node_1']['label']}:{e['node_1']['name']}"
        n2 = f"{e['node_2']['label']}:{e['node_2']['name']}"
        rel = e["relationship"]
        G_local.add_node(n1); G_local.add_node(n2)
        G_local.add_edge(n1, n2, relationship=rel)
    return all_edges_local, G_local

def draw_graph_image(G_local) -> Image.Image:
    """Render the graph to a PIL image (no blocking plt.show)."""
    fig = plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_local, k=0.85, seed=42)
    nx.draw_networkx_nodes(G_local, pos, node_size=950)
    nx.draw_networkx_labels(G_local, pos, font_size=9)
    nx.draw_networkx_edges(G_local, pos, arrows=True)
    if G_local.number_of_edges() > 0:
        nx.draw_networkx_edge_labels(
            G_local, pos,
            edge_labels={(u, v): d['relationship'] for u, v, d in G_local.edges(data=True)},
            font_size=8
        )
    plt.axis('off')
    plt.title("Knowledge Graph from Legal Document — Ontology Aligned (LLM-only)")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def edges_preview_text(edges, max_items=12) -> str:
    """Compact, readable preview string for the right-hand panel."""
    lines = []
    for e in edges[:max_items]:
        n1 = f"{e['node_1']['label']}:{e['node_1']['name']}"
        n2 = f"{e['node_2']['label']}:{e['node_2']['name']}"
        rel = e['relationship']
        desc = e.get('description')
        line = f"• {n1}  --[{rel}]->  {n2}"
        if desc:
            line += f"\n    - {textwrap.shorten(desc, width=140)}"
        lines.append(line)
    if len(edges) > max_items:
        lines.append(f"\n… and {len(edges) - max_items} more edges.")
    return "\n".join(lines) if lines else "(No edges found.)"

def _save_edges_json(edges):
    os.makedirs("/content/graph_outputs", exist_ok=True)
    out_id = uuid.uuid4().hex[:8]
    json_path = f"/content/graph_outputs/edges_{out_id}.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(edges, jf, ensure_ascii=False, indent=2)
    return json_path

# --- Handlers for the three tabs ---

def handle_txt_upload(file_obj):
    """
    Input: Uploaded .txt file
    Output: (graph image, preview string, downloadable JSON file)
    """
    if file_obj is None:
        return None, "Please upload a .txt file.", None

    with open(file_obj.name, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text.strip():
        return None, "The uploaded .txt is empty.", None

    edges, G_local = build_graph_from_text(text)
    img = draw_graph_image(G_local)
    json_path = _save_edges_json(edges)
    preview = edges_preview_text(edges, max_items=12)
    return img, preview, json_path

def handle_paste_text(text: str):
    """
    Input: Raw text from textbox
    Output: (graph image, preview string, downloadable JSON file)
    """
    if text is None or not text.strip():
        return None, "Please paste some text.", None

    edges, G_local = build_graph_from_text(text)
    img = draw_graph_image(G_local)
    json_path = _save_edges_json(edges)
    preview = edges_preview_text(edges, max_items=12)
    return img, preview, json_path

def _extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        reader = PdfReader(filepath)
        texts = []
        for page in reader.pages:
            # extract_text can return None; guard it
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        return ""

def handle_pdf_upload(file_obj):
    """
    Input: Uploaded .pdf
    Output: (graph image, preview string, downloadable JSON file)
    """
    if file_obj is None:
        return None, "Please upload a .pdf file.", None

    pdf_text = _extract_text_from_pdf(file_obj.name)
    if not pdf_text:
        return None, "Couldn't extract text from the PDF (it may be scanned or image-based).", None

    edges, G_local = build_graph_from_text(pdf_text)
    img = draw_graph_image(G_local)
    json_path = _save_edges_json(edges)
    preview = edges_preview_text(edges, max_items=12)
    return img, preview, json_path

# --- Gradio UI with 3 tabs ---
with gr.Blocks(title="Legal KG Extractor") as demo:
    gr.Markdown(
        """
        # Legal Knowledge Graph Extractor (LLM-only, Ontology-Aligned)
        Choose one of the tabs below:
        - **TXT Upload**: upload a `.txt` legal document.
        - **Paste Text**: paste raw legal text.
        - **PDF Upload**: upload a `.pdf` (selectable text PDFs work best).
        The app extracts ontology-aligned edges and renders a knowledge graph image.
        You can also download the raw **edges.json**.
        """
    )
    with gr.Tabs():
        # --- Tab 1: TXT Upload ---
        with gr.Tab("TXT Upload"):
            with gr.Row():
                with gr.Column():
                    in_txt = gr.File(label="Upload .txt", file_types=[".txt"])
                    btn_txt = gr.Button("Extract Graph from .txt")
                with gr.Column():
                    out_img_txt = gr.Image(label="Graph Image", type="pil")
                    out_preview_txt = gr.Textbox(label="Edge Preview", lines=16)
                    out_json_txt = gr.File(label="Download edges.json")
            btn_txt.click(
                fn=handle_txt_upload,
                inputs=[in_txt],
                outputs=[out_img_txt, out_preview_txt, out_json_txt]
            )

        # --- Tab 2: Paste Text ---
        with gr.Tab("Paste Text"):
            with gr.Row():
                with gr.Column():
                    in_text = gr.Textbox(
                        label="Paste legal text here",
                        lines=14,
                        placeholder="Paste your legal document text…"
                    )
                    btn_text = gr.Button("Extract Graph from Pasted Text")
                with gr.Column():
                    out_img_text = gr.Image(label="Graph Image", type="pil")
                    out_preview_text = gr.Textbox(label="Edge Preview", lines=16)
                    out_json_text = gr.File(label="Download edges.json")
            btn_text.click(
                fn=handle_paste_text,
                inputs=[in_text],
                outputs=[out_img_text, out_preview_text, out_json_text]
            )

        # --- Tab 3: PDF Upload ---
        with gr.Tab("PDF Upload"):
            with gr.Row():
                with gr.Column():
                    in_pdf = gr.File(label="Upload .pdf", file_types=[".pdf"])
                    btn_pdf = gr.Button("Extract Graph from .pdf")
                with gr.Column():
                    out_img_pdf = gr.Image(label="Graph Image", type="pil")
                    out_preview_pdf = gr.Textbox(label="Edge Preview", lines=16)
                    out_json_pdf = gr.File(label="Download edges.json")
            btn_pdf.click(
                fn=handle_pdf_upload,
                inputs=[in_pdf],
                outputs=[out_img_pdf, out_preview_pdf, out_json_pdf]
            )
            
# Launch Gradio (in Colab, it shows a public/share link & an inline iframe)
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    share=False,
    show_error=True
)