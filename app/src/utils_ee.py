"""Module for ee-related functionalities."""
import ee
import streamlit as st
from ee import oauth
from google.oauth2 import service_account
from src.utils import is_app_on_streamlit_cloud

# -------------------------------------------------------------------------
# CONFIGURATION: Set your Google Cloud Project ID here for local runs
# Or ensure it is in your .streamlit/secrets.toml file as "ee_project_id"
# -------------------------------------------------------------------------
LOCAL_PROJECT_ID = "mapaction-piet" 

@st.cache_data(show_spinner="Initializing Google Earth Engine...")
def ee_initialize(force_use_service_account: bool = False):
    """Initialise Google Earth Engine.

    Checks deployment status and initializes GEE with the correct project context.
    """
    
    # --- CASE 1: Streamlit Cloud (or forced Service Account) ---
    if force_use_service_account or is_app_on_streamlit_cloud():
        try:
            service_account_keys = st.secrets["ee_keys"]
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                service_account_keys, scopes=oauth.SCOPES
            )
            
            # FIX: Extract the project ID dynamically from the keys
            # The JSON key file usually contains a "project_id" field
            cloud_project_id = service_account_keys.get("project_id")

            # Initialize with both credentials AND project
            ee.Initialize(credentials, project=cloud_project_id)
            
        except Exception as e:
            st.error(f"Failed to authenticate with Service Account: {e}")
            raise e

    # --- CASE 2: Local Development (Personal Account) ---
    else:
        try:
            # FIX: Try to get project ID from secrets, fallback to constant
            try:
                project_id = st.secrets.get("ee_project_id", LOCAL_PROJECT_ID)
            except FileNotFoundError:
                project_id = LOCAL_PROJECT_ID

            # Initialize with the mandatory 'project' argument
            ee.Initialize(project=project_id)
            
        except ee.EEException as e:
            # Helpful error if the user hasn't authenticated locally yet
            st.error(
                "Earth Engine authentication failed. "
                "Have you run `earthengine authenticate` in your terminal "
                "and set your Project ID?"
            )
            st.stop()