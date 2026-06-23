import os
from PIL import Image as PILImage
from agno.agent import Agent, RunOutput
from agno.models.google.gemini import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-1.5-flash"

# Model options with free-tier labels
MODEL_OPTIONS = {
    "gemini-1.5-flash":      "gemini-1.5-flash  ✅ Free tier — recommended",
    "gemini-1.5-pro":        "gemini-1.5-pro    ✅ Free tier (lower quota)",
    "gemini-2.0-flash":      "gemini-2.0-flash  ✅ Free tier",
    "gemini-2.5-flash":      "gemini-2.5-flash  ✅ Free tier",
    "gemini-2.5-pro":        "gemini-2.5-pro    ⚠️ Paid tier only",
}

with st.sidebar:
    st.title("ℹ️ Configuration")
    
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

    st.markdown("---")
    st.subheader("🤖 Model Selection")
    selected_label = st.selectbox(
        "Choose Gemini model:",
        options=list(MODEL_OPTIONS.values()),
        index=0,
        help="gemini-1.5-flash has the most generous free quota. "
             "gemini-2.5-pro requires a paid plan."
    )
    # Map label back to model id
    st.session_state.selected_model = [k for k, v in MODEL_OPTIONS.items()
                                        if v == selected_label][0]
    if "2.5-pro" in st.session_state.selected_model:
        st.warning("⚠️ gemini-2.5-pro requires billing enabled on your Google account.")

    st.markdown("---")
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

def make_agent(model_id: str):
    return Agent(
        model=Gemini(
            id=model_id,
            api_key=st.session_state.GOOGLE_API_KEY
        ),
        tools=[DuckDuckGoTools()],
        markdown=True
    )

medical_agent = make_agent(st.session_state.selected_model) \
    if st.session_state.GOOGLE_API_KEY else None

if not medical_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Uploaded Medical Image",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            # Fallback chain: try selected model first, then cheaper alternatives
            FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
            current_model = st.session_state.selected_model
            models_to_try = [current_model] + [m for m in FALLBACK_MODELS if m != current_model]

            temp_path = "temp_resized_image.png"
            resized_image.save(temp_path)
            agno_image = AgnoImage(filepath=temp_path)

            success = False
            for model_id in models_to_try:
                spinner_msg = (
                    f"🔄 Analyzing with {model_id}..." if model_id == current_model
                    else f"⚡ Quota exceeded on previous model — retrying with {model_id}..."
                )
                with st.spinner(spinner_msg):
                    try:
                        agent = make_agent(model_id)
                        response: RunOutput = agent.run(query, images=[agno_image])
                        
                        if model_id != current_model:
                            st.info(
                                f"ℹ️ Used **{model_id}** (quota exceeded on {current_model}). "
                                f"Switch your model in the sidebar to avoid this."
                            )
                        st.markdown("### 📋 Analysis Results")
                        st.markdown("---")
                        st.markdown(response.content)
                        st.markdown("---")
                        st.caption(
                            "Note: This analysis is generated by AI and should be reviewed by "
                            "a qualified healthcare professional."
                        )
                        success = True
                        break  # Stop trying after first success

                    except Exception as e:
                        err_str = str(e)
                        # Check if it's a quota/rate-limit error → try next model
                        if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED", "quota"]):
                            st.warning(
                                f"⚠️ Quota exceeded for **{model_id}**. "
                                + ("Trying fallback model..." if model_id != models_to_try[-1]
                                   else "All models exhausted.")
                            )
                            continue  # Try next model
                        else:
                            # Non-quota error — show it and stop
                            st.error(f"Analysis error: {err_str}")
                            break

            if not success and models_to_try:
                st.error(
                    "❌ All models hit quota limits. Please:\n"
                    "1. **Wait ~1 minute** and try again (per-minute quota resets)\n"
                    "2. **Enable billing** on your Google Cloud project for higher limits\n"
                    "3. Check your quota at: https://ai.google.dev/gemini-api/docs/rate-limits"
                )
else:
    st.info("👆 Please upload a medical image to begin analysis")
