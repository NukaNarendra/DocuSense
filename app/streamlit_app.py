import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import DocuSensePipeline

st.set_page_config(
    page_title="DocuSense AI",
    page_icon="📚",
    layout="centered"
)


@st.cache_resource(show_spinner="Booting up DocuSense Pipeline (Vector DB & LLM)...")
def load_pipeline():
    return DocuSensePipeline()


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to initialize pipeline. Check your API keys. Error: {str(e)}")
    st.stop()

st.title("📚 DocuSense")
st.markdown("**Self-Correcting Developer Docs Assistant**")
st.markdown(
    "Ask technical questions. DocuSense will read the source documentation, expand your query, correct bad searches, and cite its sources.")
st.divider()

available_libraries = ["fastapi", "scikit-learn", "pytorch", "tensorflow", "langchain"]
selected_library = st.selectbox("Select Documentation Library", available_libraries)

user_question = st.text_input("What do you want to know?", placeholder="e.g., How do I evaluate a Random Forest model?")

if st.button("Ask DocuSense", type="primary"):
    if not user_question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner(f"Searching {selected_library.upper()} docs and generating answer..."):
            try:
                results = pipeline.run(question=user_question, target_library=selected_library)

                st.subheader("Answer")
                st.markdown(results["answer"])

                st.divider()
                with st.expander("🔍 View Pipeline Diagnostics (CRAG & Multi-Query)"):

                    if results.get("crag_triggered"):
                        st.error(
                            "🚨 **CRAG Intervened!** The initial search was too vague. The LLM Judge rejected the context and rewrote your query.")
                    else:
                        st.success("✅ **Initial Search Successful.** No CRAG correction needed.")

                    st.markdown("**Search Queries Executed (Multi-Query Expansion):**")
                    for i, q in enumerate(results["queries_executed"], 1):
                        st.markdown(f"{i}. `{q}`")

                    st.markdown(f"**Unique Documents Retrieved:** {len(results['retrieved_context'])}")
                    st.markdown(f"**Execution Time:** {results['execution_time']:.2f} seconds")

            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")