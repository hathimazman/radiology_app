import os
import streamlit as st
import base64
import io
import uuid
from datetime import datetime

# Set page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Radiology AI Learning Platform", layout="wide")

# Set environment variable to disable Streamlit watchdog
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Import other libraries after setting page config
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# Import Firestore libraries
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

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

# Initialize Firebase app if not already initialized
@st.cache_resource
def initialize_firebase():
    try:
        # Check if the app is already initialized
        firebase_admin.get_app()
    except ValueError:
        # Initialize the app with your credentials file
        # You need to create a service account key from Firebase console and save it
        cred = credentials.Certificate("radiology-app-firebase-key.json")  # Path to your credentials file
        firebase_admin.initialize_app(cred)
    
    # Return Firestore client
    return firestore.client()

# Initialize the model at startup, not during refresh cycles
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Firebase and Firestore
try:
    db = initialize_firebase()
    st.sidebar.success("Connected to Firestore!")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Firestore: {e}")
    st.stop()

# Load NLP Model
model = load_model()

# Function to retrieve random X-ray case
def get_random_case():
    try:
        # Get all documents from cases collection
        cases_ref = db.collection('radiology_cases')
        cases = cases_ref.stream()
        
        # Convert to list
        cases_list = list(cases)
        
        # If no cases found
        if not cases_list:
            st.warning("No cases found in the database. Please add some cases first.")
            return None
        
        # Select a random case
        import random
        random_case = random.choice(cases_list)
        case_data = random_case.to_dict()
        
        # Add case ID to the data
        case_data['id'] = random_case.id
        
        # Convert base64 image string back to bytes
        image_bytes = base64.b64decode(case_data['radiological_image'])
        
        # Return in format similar to previous PostgreSQL function
        return (
            case_data['id'],
            image_bytes,
            case_data['image_description'],
            case_data['diagnosis'],
            case_data['management']
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
    
    # Calculate similarity
    similarity = util.pytorch_cos_sim(
        model.encode(student_answer, convert_to_tensor=True),
        model.encode(expert_answer, convert_to_tensor=True)
    ).item()
    
    feedback = "Good job!" if similarity > 0.85 else "Consider revising."
    if missing_keywords:
        # Limit to top 5 missing keywords to avoid overwhelming feedback
        top_missing = list(missing_keywords)[:5]
        feedback += f" Missing key terms: {', '.join(top_missing)}."
        if len(missing_keywords) > 5:
            feedback += " (and others)"
    
    return similarity, feedback

# Streamlit UI
st.title("Radiology AI Learning Platform")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Landing Page", "📚 Student Exercise", "➕ Add X-ray Case", "⚙️ Database Migration"])

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
    6. If you're migrating from PostgreSQL, use the **Database Migration** tab
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
            image = Image.open(io.BytesIO(image_data))
            
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
                st.image(image, caption="Assess this X-ray", use_column_width=False, width=min(image.width, max_width))
            
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
                    
                # Optional: Save student responses to Firestore
                if st.button("Save My Response for Review"):
                    try:
                        # Create a new document in the student_responses collection
                        student_response = {
                            'case_id': case_id,
                            'student_description': student_description,
                            'student_diagnosis': student_diagnosis,
                            'student_management': student_management,
                            'description_score': desc_score,
                            'diagnosis_score': diag_score,
                            'management_score': mgmt_score,
                            'timestamp': firestore.SERVER_TIMESTAMP
                        }
                        
                        db.collection('student_responses').add(student_response)
                        st.success("Your response has been saved for review!")
                    except Exception as e:
                        st.error(f"Error saving response: {e}")
                
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Click 'Get a Random X-ray' to start a new assessment.")

# Tab 3: Add More X-ray Cases
with tab3:
    st.header("Add a New Radiology Case")
    
    # Set a passcode for verification
    ADMIN_PASSCODE = "rad2025"  # You can change this to your desired passcode
    
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
            
            st.image(preview_img, caption="Image Preview", use_column_width=False)
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
                
                # Convert image to bytes and then to base64 for Firestore storage
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
                img_bytes = img_byte_arr.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # Create a new document in Firestore
                case_data = {
                    'radiological_image': img_base64,
                    'image_description': description,
                    'diagnosis': diagnosis,
                    'management': management,
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'created_by': 'admin'  # You could replace this with a user ID if you have authentication
                }
                
                # Add the document to the 'radiology_cases' collection
                db.collection('radiology_cases').add(case_data)
                
                st.success("Case added successfully to Firestore!")
            except Exception as e:
                st.error(f"Error saving case: {e}")
        else:
            st.warning("Please fill in all fields.")

# Tab 4: Database Migration Tool
with tab4:
    st.header("PostgreSQL to Firestore Migration")
    st.info("This tool helps migrate your existing PostgreSQL database to Firestore.")
    
    # Database connection settings
    st.subheader("PostgreSQL Connection Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        pg_dbname = st.text_input("Database Name", "radiology_app")
        pg_user = st.text_input("Database User", "postgres")
        pg_password = st.text_input("Database Password", "hathimukmfper2025", type="password")
    
    with col2:
        pg_host = st.text_input("Database Host", "localhost")
        pg_port = st.text_input("Database Port", "5432")
        pg_table = st.text_input("Table Name", "radiology_images")
    
    # Add passcode for migration
    migration_passcode = st.text_input("Enter admin passcode to migrate data", type="password")
    
    if st.button("Start Migration"):
        if not migration_passcode:
            st.warning("Please enter the admin passcode to proceed with migration.")
        elif migration_passcode != ADMIN_PASSCODE:
            st.error("Incorrect passcode. Migration aborted.")
        else:
            try:
                # Import psycopg2 here to avoid dependencies if not using migration
                import psycopg2
                
                # PostgreSQL Connection Settings
                DB_PARAMS = {
                    "dbname": pg_dbname,
                    "user": pg_user,
                    "password": pg_password,
                    "host": pg_host,
                    "port": pg_port
                }
                
                # Create progress bar
                progress_bar = st.progress(0)
                migration_status = st.empty()
                migration_status.info("Connecting to PostgreSQL database...")
                
                # Connect to PostgreSQL
                conn = psycopg2.connect(**DB_PARAMS)
                cur = conn.cursor()
                
                # Get all records from PostgreSQL
                migration_status.info("Fetching records from PostgreSQL...")
                cur.execute(f"SELECT id, radiological_image, image_description, diagnosis, management FROM {pg_table};")
                records = cur.fetchall()
                
                if not records:
                    migration_status.warning("No records found in PostgreSQL database.")
                else:
                    migration_status.info(f"Found {len(records)} records. Starting migration...")
                    
                    # Create a batch writer for Firestore
                    batch = db.batch()
                    
                    # Process each record
                    for i, record in enumerate(records):
                        record_id, image_data, description, diagnosis, management = record
                        
                        # Convert binary image data to base64
                        img_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # Create new document reference
                        doc_ref = db.collection('radiology_cases').document()
                        
                        # Prepare data
                        case_data = {
                            'original_pg_id': record_id,
                            'radiological_image': img_base64,
                            'image_description': description,
                            'diagnosis': diagnosis,
                            'management': management,
                            'timestamp': firestore.SERVER_TIMESTAMP,
                            'migrated_at': datetime.now(),
                            'source': 'postgresql_migration'
                        }
                        
                        # Add to batch
                        batch.set(doc_ref, case_data)
                        
                        # Update progress
                        progress = int((i + 1) / len(records) * 100)
                        progress_bar.progress(progress)
                        migration_status.info(f"Migrating record {i+1} of {len(records)}...")
                        
                        # Commit batch every 500 records (Firestore limit)
                        if (i + 1) % 500 == 0:
                            batch.commit()
                            batch = db.batch()
                    
                    # Commit any remaining records
                    batch.commit()
                    
                    # Complete
                    progress_bar.progress(100)
                    migration_status.success(f"Migration complete! {len(records)} records migrated to Firestore.")
                
                # Close PostgreSQL connection
                cur.close()
                conn.close()
                
            except Exception as e:
                st.error(f"Migration failed: {e}")