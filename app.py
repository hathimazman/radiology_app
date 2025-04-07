import os
import streamlit as st
from PIL import Image
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import supabase
from dotenv import load_dotenv
import time
import logging

# Only load .env file for local development, not needed in Streamlit Cloud
if not os.path.exists('.streamlit/secrets.toml'):
    from dotenv import load_dotenv
    load_dotenv()

# Set page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Radiology AI Learning Platform", layout="wide")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manually disable torch watchdog monitoring to prevent runtime errors
import sys
if 'torch' in sys.modules:
    import torch
    if hasattr(torch, '_C') and hasattr(torch._C, '_log_api_usage_once'):
        # Patch the function that's causing the error
        torch._C._log_api_usage_once = lambda *args, **kwargs: None

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Supabase connection
@st.cache_resource
def init_connection():
    # Get Supabase credentials from environment variables
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        return None
    
    try:
        client = supabase.create_client(supabase_url, supabase_key)
        return client
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        return None

# Initialize the model at startup, not during refresh cycles
@st.cache_resource
def load_model():
    try:
        # Try to import and load the model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Successfully loaded SentenceTransformer model")
            return {"model": model, "type": "transformer"}
        except Exception as e:
            logger.warning(f"Error loading SentenceTransformer model: {e}")
            logger.info("Falling back to basic comparison method")
            return {"model": None, "type": "basic"}
    except Exception as e:
        logger.error(f"Failed to initialize any model: {e}")
        return {"model": None, "type": "basic"}

# Load NLP Model
model_info = load_model()
model_type = model_info["type"]
model = model_info["model"]

# Show a message if using basic comparison instead of the AI model
if model_type == "basic":
    st.warning("Using basic text comparison instead of AI model due to API rate limits. Refresh in a few minutes to try again.")

# Function to retrieve random X-ray case
def get_random_case():
    client = init_connection()
    if not client:
        return None
    
    try:
        response = client.table('radiology_images').select('*').limit(100).execute()
        
        # Check if we got any data
        if not response.data:
            st.error("No radiology images found in the database.")
            return None
            
        # Select a random case from the results
        import random
        case = random.choice(response.data)
        
        # Extract the image data
        image_url = case.get('radiological_image')
        
        # If image is stored as a URL in Supabase Storage
        if image_url and isinstance(image_url, str) and image_url.startswith('http'):
            import requests
            response = requests.get(image_url)
            image_data = response.content
        else:
            # If image is stored directly in the database (base64 or bytea)
            image_data = case.get('radiological_image')
            
        return (
            case.get('id'), 
            image_data, 
            case.get('image_description'), 
            case.get('diagnosis'), 
            case.get('management')
        )
    except Exception as e:
        st.error(f"Error retrieving case: {e}")
        return None

# Function to evaluate student answer
def evaluate_answer(student_answer, expert_answer):
    if not student_answer or not expert_answer:
        return 0.0, "Please provide an answer."
    
    student_words = set(word_tokenize(student_answer.lower())) - set(stopwords.words("english"))
    expert_words = set(word_tokenize(expert_answer.lower())) - set(stopwords.words("english"))
    missing_keywords = expert_words - student_words
    
    # Calculate similarity based on available model
    if model_type == "transformer":
        try:
            from sentence_transformers import util
            similarity = util.pytorch_cos_sim(
                model.encode(student_answer, convert_to_tensor=True),
                model.encode(expert_answer, convert_to_tensor=True)
            ).item()
        except Exception as e:
            logger.error(f"Error using transformer model: {e}")
            # Fallback to basic comparison
            common_words = student_words.intersection(expert_words)
            similarity = len(common_words) / len(expert_words) if expert_words else 0
    else:
        # Basic comparison using word overlap
        common_words = student_words.intersection(expert_words)
        similarity = len(common_words) / len(expert_words) if expert_words else 0
    
    feedback = "Good job!" if similarity > 0.7 else "Consider revising."
    if missing_keywords:
        # Limit to top 5 missing keywords to avoid overwhelming feedback
        top_missing = list(missing_keywords)[:5]
        feedback += f" Missing key terms: {', '.join(top_missing)}."
        if len(missing_keywords) > 5:
            feedback += " (and others)"
    
    return similarity, feedback

# Streamlit UI
st.title("Radiology AI Learning Platform")

tab1, tab2, tab3 = st.tabs(["🏠 Landing Page", "📚 Student Exercise", "➕ Add X-ray Case"])

# Tab 1: Landing Page
with tab1:
    st.header("Welcome to the Radiology Learning System")
    st.write("This platform helps medical students assess their radiology interpretation skills.")
    
    # Add more content to the landing page
    st.subheader("How to use this platform")
    st.markdown("""
    1. Go to the **Student Exercise** tab to practice your radiology skills
    2. Click "Get a Random X-ray" to load a case
    3. Analyze the image and provide your findings
    4. Submit your answer for AI-based feedback
    5. Faculty members can add new cases in the **Add X-ray Case** tab
    """)

# Tab 2: Student Exercise
with tab2:
    st.header("Radiology Assessment")
    
    # Use session state to persist the case between reruns
    if 'current_case' not in st.session_state:
        st.session_state.current_case = None
        st.session_state.submitted = False
    
    if st.button("Get a Random X-ray"):
        st.session_state.current_case = get_random_case()
        st.session_state.submitted = False
        
    if st.session_state.current_case:
        case_id, image_data, expert_description, expert_diagnosis, expert_management = st.session_state.current_case
        
        try:
            # Handle different types of image data
            if isinstance(image_data, str) and image_data.startswith('http'):
                # If it's a URL
                import requests
                response = requests.get(image_data)
                image = Image.open(io.BytesIO(response.content))
            elif isinstance(image_data, bytes):
                # If it's bytes
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str) and image_data.startswith('data:image'):
                # If it's a data URI
                import base64
                image_data = image_data.split(',')[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            elif isinstance(image_data, str):
                # Try to decode as base64
                try:
                    import base64
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                except:
                    st.error("Unable to decode image data")
                    image = None
            else:
                # Handle other potential formats from Supabase
                st.error("Unsupported image format")
                image = None
            
            if image:
                # Resize image while maintaining aspect ratio
                max_width = 800  # Set maximum width
                if image.width > max_width:
                    ratio = max_width / image.width
                    new_height = int(image.height * ratio)
                    image = image.resize((max_width, new_height), Image.LANCZOS)
                
                # Create columns to control image width
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Display image with specific width in the center column (no auto scaling)
                    st.image(image, caption="Assess this X-ray", width=min(image.width, max_width))
                
                # Student input fields
                student_description = st.text_area("Describe the X-ray findings:")
                student_diagnosis = st.text_input("Diagnosis:")
                student_management = st.text_area("Management Plan:")
                
                if st.button("Submit Answer") or st.session_state.submitted:
                    st.session_state.submitted = True
                    
                    desc_score, desc_feedback = evaluate_answer(student_description, expert_description)
                    diag_score, diag_feedback = evaluate_answer(student_diagnosis, expert_diagnosis)
                    mgmt_score, mgmt_feedback = evaluate_answer(student_management, expert_management)
                    
                    st.write(f"**Description Score:** {desc_score:.2f} - {desc_feedback}")
                    st.write(f"**Diagnosis Score:** {diag_score:.2f} - {diag_feedback}")
                    st.write(f"**Management Score:** {mgmt_score:.2f} - {mgmt_feedback}")
                    
                    # Add expert answer reveal option
                    if st.checkbox("Show expert answers"):
                        st.info("**Expert Description:**\n" + expert_description)
                        st.info("**Expert Diagnosis:**\n" + expert_diagnosis)
                        st.info("**Expert Management:**\n" + expert_management)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Click 'Get a Random X-ray' to start a new assessment.")

# Tab 3: Add More X-ray Cases
with tab3:
    st.header("Add a New Radiology Case")
    
    # Set a passcode for verification
    ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "rad2025")  # Get from environment variable or use default
    
    uploaded_image = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        # Preview the uploaded image with resizing
        try:
            preview_img = Image.open(uploaded_image)
            preview_img.verify()  # Verify it's a valid image
            uploaded_image.seek(0)  # Reset file pointer
            
            # Resize preview image
            preview_img = Image.open(uploaded_image)
            max_width = 400  # Smaller preview size
            if preview_img.width > max_width:
                ratio = max_width / preview_img.width
                new_height = int(preview_img.height * ratio)
                preview_img = preview_img.resize((max_width, new_height), Image.LANCZOS)
            
            st.image(preview_img, caption="Image Preview", width=max_width)
            uploaded_image.seek(0)  # Reset file pointer again after preview
        except Exception as e:
            st.error(f"Error previewing image: {e}")
    
    description = st.text_area("Expert Image Description")
    diagnosis = st.text_input("Expert Diagnosis")
    management = st.text_area("Expert Management Plan")
    
    # Add passcode verification
    passcode = st.text_input("Enter admin passcode to save", type="password")
    
    if st.button("Save Case"):
        if not passcode:
            st.warning("Please enter the admin passcode to proceed.")
        elif passcode != ADMIN_PASSCODE:
            st.error("Incorrect passcode. Please try again.")
        elif uploaded_image and description and diagnosis and management:
            try:
                # Validate image
                try:
                    img = Image.open(uploaded_image)
                    img.verify()  # Verify that it's a valid image
                    uploaded_image.seek(0)  # Reset file pointer
                except Exception:
                    st.error("Invalid image file. Please upload a valid image.")
                    st.stop()
                
                # Process the image - resize if needed before saving to database
                img = Image.open(uploaded_image)
                max_size = (1200, 1200)  # Maximum dimensions for stored images
                img.thumbnail(max_size, Image.LANCZOS)
                
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Initialize Supabase client
                client = init_connection()
                if client:
                    # Upload image to Supabase Storage
                    import uuid
                    import base64
                    
                    # Option 1: Store as base64 directly in the database
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Insert record into the database
                    response = client.table('radiology_images').insert({
                        'radiological_image': img_base64,  # Store as base64
                        'image_description': description,
                        'diagnosis': diagnosis,
                        'management': management
                    }).execute()
                    
                    if hasattr(response, 'error') and response.error:
                        st.error(f"Error saving to Supabase: {response.error}")
                    else:
                        st.success("Case added successfully!")
            except Exception as e:
                st.error(f"Error saving case: {e}")
        else:
            st.warning("Please fill in all fields.")
