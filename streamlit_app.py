import streamlit as st
import pandas as pd
import tempfile
import os
from fpdf import FPDF

# Gemini AI imports
import base64
from google import genai
from google.genai import types

# Supabase imports
from supabase import create_client

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Invision Insolvency Intelligence Solutions", layout="wide")
st.title("Invision Insolvency Intelligence Solutions")

# --------- UTILITY FUNCTIONS --------------

def call_gemini_ai(input_text):
    """
    Calls Gemini AI (Google GenAI) using google-genai python package.
    Requires:
      - pip install google-genai
      - GEMINI_API_KEY in Streamlit secrets or env
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            return "No API key found in secrets or environment variable."

    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=input_text),
                ],
            ),
        ]
        tools = [
            types.Tool(url_context=types.UrlContext()),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )
        out = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            out += chunk.text or ""
        return out
    except Exception as e:
        return f"Failed to call Gemini AI: {str(e)}"

def extract_text_from_file(uploaded_file):
    # Basic extractor: text, pdf, docx, images (optional: add more robust extractors as needed)
    import mimetypes
    file_type = mimetypes.guess_type(uploaded_file.name)[0] or ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext in [".txt"]:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception:
                text = "Could not extract PDF text."
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                text = "Could not extract DOCX text."
        elif "image" in file_type or ext in [".png", ".jpg", ".jpeg"]:
            try:
                import pytesseract
                from PIL import Image
                img = Image.open(uploaded_file)
                text = pytesseract.image_to_string(img)
            except Exception:
                text = "Could not extract image text."
        else:
            text = "Unsupported file type."
    except Exception:
        text = "Failed to process file."
    return text[:6000] # Limit input for AI

def assets_template_csv():
    return (
        "Asset Name,Category,Location,Original Value,Date of Acquisition,Depreciation Rate,Current Condition,Comparable Assets Info\n"
        "Machinery,Plant & Machinery,Plant A,1000000,2018-03-01,10%,Good,Similar Machinery at Plant B\n"
        "Building,Real Estate,Factory Campus,5000000,2016-08-15,5%,Needs Repair,Industrial Building in same area\n"
        "Vehicle,Transport,Warehouse,700000,2019-10-23,15%,Fair,Truck Model Z as per market\n"
    )

def generate_pdf(text, title="Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.seek(0)
        pdf_bytes = tmp_pdf.read()
    return pdf_bytes

# ------------------- SUPABASE DB UTILITY -------------------
@st.cache_resource(show_spinner=False)
def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        st.error("Missing Supabase URL or Key in secrets!")
        return None
    return create_client(url, key)

def run_supabase_sql(query):
    client = get_supabase_client()
    if not client:
        return None, "Supabase connection not available."
    try:
        result = client.rpc('run_sql', {'sql': query}).execute() # Use run_sql function if created on Supabase
        if result.data:
            df = pd.DataFrame(result.data)
            return df, None
        else:
            return pd.DataFrame(), None
    except Exception as e:
        return None, str(e)

def fetch_nclt_table_data(search_text=None):
    client = get_supabase_client()
    if not client:
        return None, "Supabase connection not available."
    try:
        query = client.table("nclt_intelligence").select("*")
        if search_text:
            # Attempt basic search: check all columns for LIKE match
            meta = client.table("nclt_intelligence").select("*").limit(1).execute()
            if meta.data:
                columns = list(meta.data[0].keys())
                # Build OR filters for LIKE match
                search_filter = "|".join([f"{col}.ilike.%{search_text}%" for col in columns])
                query = query.or_(search_filter)
        result = query.execute()
        if result.data:
            df = pd.DataFrame(result.data)
            return df, None
        else:
            return pd.DataFrame(), None
    except Exception as e:
        return None, str(e)

# ------------------- APP LAYOUT -------------------

tab1, tab2, tab3 = st.tabs([
    "Document Intelligence",
    "Valuation Reports & Compliance",
    "NCLT Cases"
])

# ----------------- TAB 1: DOCUMENT INTELLIGENCE -----------------
with tab1:
    st.header("Document Intelligence")
    doc_type = st.selectbox(
        "Select Document Type",
        ["Legal Document", "IBC Document", "Invoice", "Other"]
    )
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, DOCX, TXT, PNG, JPG)", 
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], 
        key="doc_upload"
    )
    if uploaded_file:
        st.info(f"Uploaded: {uploaded_file.name}")
        extracted_text = extract_text_from_file(uploaded_file)
        st.text_area("Extracted Text (preview)", value=extracted_text[:1500], height=200)
        if st.button("Analyze with AI"):
            with st.spinner("Analyzing document with Gemini AI..."):
                prompt = (
                    f"This is a {doc_type} for insolvency/intelligence context. "
                    f"Please perform a comprehensive audit/analysis and generate a detailed professional report:\n\n"
                    f"{extracted_text}"
                )
                analysis = call_gemini_ai(prompt)
            st.success("AI Analysis Complete")
            st.subheader("Comprehensive Audit/Analysis Report")
            st.text_area("AI Report", value=analysis, height=300)
            st.download_button(
                "Download Analysis Report (TXT)",
                data=analysis,
                file_name="analysis_report.txt",
                mime="text/plain"
            )
            pdf_bytes = generate_pdf(analysis, title="Document Intelligence Report")
            st.download_button(
                "Download Analysis Report (PDF)",
                data=pdf_bytes,
                file_name="analysis_report.pdf",
                mime="application/pdf"
            )

# ----------------- TAB 2: VALUATION REPORT & COMPLIANCE -----------------
with tab2:
    st.header("Valuation Reports & Compliance")
    col1, col2 = st.columns([2, 1])
    with col1:
        asset_file = st.file_uploader(
            "Upload Asset CSV (use template below)", 
            type="csv", key="asset_csv"
        )
    with col2:
        st.markdown("**CSV Template**")
        st.download_button(
            "Download Asset CSV Template",
            data=assets_template_csv(),
            file_name="assets_template.csv",
            mime="text/csv"
        )

    if asset_file:
        try:
            df = pd.read_csv(asset_file)
            st.success("Assets loaded successfully!")
            st.dataframe(df.head(20))
            if st.button("Run IBC Compliant Valuation"):
                asset_csv_text = df.to_csv(index=False)
                ai_prompt = (
                    "You are an IBC-compliant valuation expert. Given these asset details, "
                    "perform a professional, comparable-based, and financial valuation. "
                    "Give a detailed and structured valuation report suitable for compliance, "
                    "listing assumptions, comparable analysis, and rationale. The data:\n\n"
                    f"{asset_csv_text}"
                )
                with st.spinner("Performing Gemini AI-powered valuation..."):
                    valuation_report = call_gemini_ai(ai_prompt)
                st.success("Valuation Complete")
                st.subheader("IBC Compliant Valuation Report")
                st.text_area("Valuation Report", value=valuation_report, height=300)
                st.download_button(
                    "Download Valuation Report (TXT)",
                    data=valuation_report,
                    file_name="valuation_report.txt",
                    mime="text/plain"
                )
                pdf_bytes = generate_pdf(valuation_report, title="IBC Valuation Report")
                st.download_button(
                    "Download Valuation Report (PDF)",
                    data=pdf_bytes,
                    file_name="valuation_report.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# ----------------- TAB 3: NCLT CASES -----------------
with tab3:
    st.header("NCLT Cases Intelligence Database")
    st.markdown("> Powered by Supabase. All columns fetched from `nclt_intelligence` table.")

    # Basic Search Bar
    search_text = st.text_input("Basic Search (searches all columns):", "")
    if st.button("Search NCLT Cases"):
        with st.spinner("Searching NCLT cases..."):
            df, err = fetch_nclt_table_data(search_text if search_text else None)
        if err:
            st.error(f"Error: {err}")
        elif df is not None and not df.empty:
            st.success(f"Found {len(df)} records.")
            st.dataframe(df)
        else:
            st.warning("No results found.")
    else:
        # Initial load: show top 50 records
        df, err = fetch_nclt_table_data()
        if err:
            st.error(f"Error: {err}")
        elif df is not None and not df.empty:
            st.dataframe(df.head(50))

    # Advanced SQL Query Editor (read-only)
    st.markdown("### Advanced SQL Query (Read-only)")
    st.markdown(
        "Write custom **SELECT** queries. Example: `SELECT * FROM nclt_intelligence WHERE status = 'Active' LIMIT 10`"
    )
    sql_query = st.text_area("SQL Query", "SELECT * FROM nclt_intelligence LIMIT 20", height=80)
    if st.button("Run SQL Query"):
        if not sql_query.strip().lower().startswith("select"):
            st.warning("Only SELECT queries are allowed!")
        else:
            with st.spinner("Running SQL query..."):
                df_sql, err_sql = run_supabase_sql(sql_query)
            if err_sql:
                st.error(f"Error: {err_sql}")
            elif df_sql is not None and not df_sql.empty:
                st.success(f"Query returned {len(df_sql)} records.")
                st.dataframe(df_sql)
            else:
                st.warning("No results from query.")

# ------------------- FOOTER INFO -------------------
st.markdown("""
---
**Note:**  
- The app uses Google Gemini AI via google-genai python package.  
- Requires: `pip install streamlit pandas fpdf PyPDF2 python-docx google-genai`  
- For image text extraction: `pip install pytesseract pillow` (and install Tesseract if you want image text extraction).
- Add your API key to Streamlit secrets as `GEMINI_API_KEY` or set as environment variable.
- For NCLT tab: add `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets.  
- Supabase read-only queries require a function called `run_sql` on Supabase (Postgres) that runs arbitrary SELECT SQL and returns results.
""")
