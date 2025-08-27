import streamlit as st
import pandas as pd
import tempfile
import os
from fpdf import FPDF
from datetime import datetime

# Gemini AI imports
from google import genai
from google.genai import types

# Supabase imports
from supabase import create_client

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Invision Insolvency Intelligence Solutions", layout="wide")

# Display current time and user
current_time = "2025-08-27 17:15:39"  # UTC
current_user = "evertechno"

st.title("Invision Insolvency Intelligence Solutions")
st.markdown(f"""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px'>
    <small>
        Current Time (UTC): {current_time}<br>
        User: {current_user}
    </small>
</div>
""", unsafe_allow_html=True)

# --------- UTILITY FUNCTIONS --------------

def call_gemini_ai(input_text):
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
    return text[:6000]

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
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Generated on {current_time} UTC by {current_user}", ln=True, align="R")
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

def fetch_nclt_table_data(search_text=None):
    client = get_supabase_client()
    if not client:
        return None, "Supabase connection not available."
    try:
        if search_text:
            # Clean search_text for safety and escape single quotes
            search_text = search_text.strip().replace("'", "''")
            # Build SQL query with proper type casting and ILIKE
            sql = f"""
            SELECT * FROM nclt_intelligence 
            WHERE 
                CASE 
                    WHEN '{search_text}' ~ '^[0-9]+$' 
                    THEN "Unique ID" = {search_text}::bigint 
                    ELSE false 
                END
                OR "Unique ID"::text ILIKE '%{search_text}%'
                OR "Regulatory File Name" ILIKE '%{search_text}%'
                OR "Content" ILIKE '%{search_text}%'
            ORDER BY "Unique ID"
            """
            result = client.rpc('run_sql', {'query_text': sql}).execute()
        else:
            # Default query for initial load
            sql = 'SELECT * FROM nclt_intelligence ORDER BY "Unique ID" LIMIT 50'
            result = client.rpc('run_sql', {'query_text': sql}).execute()
            
        if result.data:
            df = pd.DataFrame(result.data)
            # Ensure column order
            df = df[["Unique ID", "Regulatory File Name", "Content"]]
            return df, None
        else:
            return pd.DataFrame(), None
    except Exception as e:
        return None, str(e)

def run_supabase_sql(query):
    client = get_supabase_client()
    if not client:
        return None, "Supabase connection not available."
    try:
        # Validate query starts with SELECT
        if not query.strip().lower().startswith("select"):
            return None, "Only SELECT queries are allowed!"
        
        # Execute query through run_sql function
        result = client.rpc('run_sql', {'query_text': query}).execute()
        
        if result.data:
            df = pd.DataFrame(result.data)
            # Ensure column order if possible
            try:
                df = df[["Unique ID", "Regulatory File Name", "Content"]]
            except:
                pass # Keep original order if columns don't match
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
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            pdf_bytes = generate_pdf(analysis, title="Document Intelligence Report")
            st.download_button(
                "Download Analysis Report (PDF)",
                data=pdf_bytes,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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
            st.dataframe(df)
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
                    file_name=f"valuation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                pdf_bytes = generate_pdf(valuation_report, title="IBC Valuation Report")
                st.download_button(
                    "Download Valuation Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"valuation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# ----------------- TAB 3: NCLT CASES -----------------
with tab3:
    st.header("NCLT Cases Intelligence Database")
    st.markdown("""
    > Powered by Supabase. All columns fetched from `nclt_intelligence` table:  
    > **Unique ID** (number), **Regulatory File Name** (text), **Content** (text)
    """)

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_text = st.text_input(
            "Basic Search (searches across all fields):",
            placeholder="Enter search term (e.g., mumbai bench, file number, or content keywords)"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_button = st.button("🔍 Search NCLT Cases", use_container_width=True)

    if search_button:
        with st.spinner("Searching NCLT cases..."):
            df, err = fetch_nclt_table_data(search_text if search_text else None)
        if err:
            st.error(f"Error: {err}")
        elif df is not None and not df.empty:
            st.success(f"Found {len(df)} records.")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No results found.")
    else:
        # Initial load: show top 50 records
        with st.spinner("Loading recent cases..."):
            df, err = fetch_nclt_table_data()
        if err:
            st.error(f"Error: {err}")
        elif df is not None and not df.empty:
            st.info("Showing 50 most recent cases")
            st.dataframe(df, use_container_width=True)

    # Advanced SQL Query Section
    st.markdown("### Advanced SQL Query (Read-only)")
    with st.expander("ℹ️ SQL Query Help"):
        st.markdown("""
        Write custom **SELECT** queries to search the NCLT database. Examples:
        ```sql
        -- Search by content
        SELECT * FROM nclt_intelligence 
        WHERE "Content" ILIKE '%mumbai bench%' 
        LIMIT 10

        -- Search by Regulatory File Name
        SELECT * FROM nclt_intelligence 
        WHERE "Regulatory File Name" ILIKE '%IBC%' 
        ORDER BY "Unique ID" DESC 
        LIMIT 20
        ```
        """)
    
    sql_query = st.text_area(
        "SQL Query",
        value='SELECT * FROM nclt_intelligence LIMIT 20',
        height=100,
        help="Write your SELECT query here. Only SELECT operations are allowed."
    )

    if st.button("▶️ Run SQL Query"):
        if not sql_query.strip().lower().startswith("select"):
            st.warning("⚠️ Only SELECT queries are allowed!")
        else:
            with st.spinner("Executing query..."):
                df_sql, err_sql = run_supabase_sql(sql_query)
            
