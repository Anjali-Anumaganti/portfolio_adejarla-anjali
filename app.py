import streamlit as st

# Set page configuration
st.set_page_config(page_title="Anjali's AI/ML Portfolio", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #ffffff, #e6f0ff);
            padding: 10px;
            border-radius: 10px;
        }
        .stApp {
            background-color: #f7f7f9;
        }
        .css-1d391kg, .css-1v0mbdj, .st-emotion-cache-1avcm0n {
            background-color: transparent !important;
            color: white;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #003366, #336699);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with profile image and contact info
st.sidebar.image("image.pnj.jpg", width=150)  # Ensure the image file is in your directory
st.sidebar.markdown("## **Adejarla Anjali**")
st.sidebar.markdown("### AI/ML Engineer")
st.sidebar.markdown("""
- 📧 [anumagantianjali1101@gmail.com](mailto:anumagantianjali1101@gmail)  
- 🐈 [GitHub](https://github.com/Anjali-Anumaganti)  
- 🔗 [LinkedIn](https://www.linkedin.com/in/anumaganti-anjali-130292272/)  
""")

# Resume download button
#with open(r"C:\Users\Shiva Kumar\Downloads\ADEJARLA_ANJALI_Naukri_Style_Resume_Trainee.docx", "rb") as file:
    #st.sidebar.download_button("📄 Download Resume", file, file_name="Anjali_Resume.pdf")

# Tabs for the main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Summary", "📌 Personal Info", "💼 Experience",
    "🧠 Skills", "📚 Education", "📜 Certifications", "📊 Projects"
])

# Summary Tab
with tab1:
    st.title("👋 Hi, I’m Adejarla Anjali")
    st.subheader("AI/ML Engineer | LLM Specialist | Data Scientist")
    st.markdown("""
    - As an AI/ML Intern, I collaborated with data science and engineering teams to contribute to the development and deployment of machine learning models and AI solutions. My key responsibilities included:

- Data Collection & Processing: Cleaned, structured, and validated datasets for ML training and analysis to ensure high data integrity and efficient workflows.

- Model Development & Evaluation: Assisted in designing, training, and testing ML models, and conducted performance evaluations for optimization.

- AI Application Optimization: Analyzed applications to enhance computational efficiency and scalability.

- Research & Experimentation: Explored latest AI/ML trends and contributed to integrating new methodologies into existing projects.

- Automation & Workflow Enhancement: Developed scripts to automate repetitive tasks and improve workflow efficiency, integrating DevOps practices where applicable.

- Documentation & Reporting: Maintained clear documentation of experiments, model metrics, and project insights to support continuous improvement.

- Team Collaboration: Worked alongside data scientists, software engineers, and DevOps teams, actively contributing to brainstorming and solution-building sessions.

- Technologies & Tools: TensorFlow, PyTorch, Git, Matplotlib, Seaborn, Cloud ML platforms.


    """)

# Personal Info Tab
with tab2:
    st.header("📌 Personal Info")
    st.markdown("""
    - **Name:** Adejarla Anjali  
    - **DOB:** 11/01/2002  
    - **Gender:** Female  
    - **Location:** Hyderabad, Telangana  
    - **Languages:** English, Telugu, Hindi  
    - **Open to:** Full-time, Contract, Research  
    """)

# Experience Tab
with tab3:
    st.header("💼 Experience")
    st.subheader("AI/ML Engineer – Lyros Technologies Pvt. Ltd")
    st.caption("📍 Cyber Towers, Hitech City, Hyderabad | 🕒 Feb 2025 – Present")
    st.markdown("""
    - Developed ML Models: Worked on supervised learning techniques including classification (e.g., breast cancer detection, spam filtering) and regression models (e.g., sales prediction, price forecasting).

- 🧠 Deep Learning: Built and fine-tuned neural networks using TensorFlow and PyTorch for tasks such as image classification and sentiment analysis.

- 🌐 NLP Applications: Implemented Natural Language Processing (NLP) workflows including tokenization, stemming, named entity recognition, and text classification.

- 🧾 LLM & Transformers: Built and optimized Large Language Models and transformer-based architectures (e.g., BERT, GPT) for document summarization and knowledge-based Q&A 

- ⚙️ MLOps Practices: Integrated workflows using MLflow, Docker, and Kubernetes for reproducibility and scalable deployment.

- 📈 Model Evaluation: Focused on performance tuning using metrics such as accuracy, precision, recall, RMSE, and AUC-ROC.

- CI/CD Deployment with Render & Supabase: Automated end-to-end deployment of ML/NLP apps using Render for hosting and Supabase for backend/database services. Integrated CI/CD pipelines for continuous testing, model updates, and seamless delivery

 
    """)

# Skills Tab
with tab4:
    st.header("🧠 Technical Skills")
    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **Core ML/DL**  
    - Python  
    - Scikit-learn  
    - PyTorch  
    - TensorFlow  
    - XGBoost / LightGBM  
    """)
    col2.markdown("""
    **Data Science & AI**
     - Machine Learning
     - Deep Learning
     - NLP
     - Transformers
     - LLMs
     - Pandas / NumPy
 
  
    """)
    col3.markdown("""
    **Deployment & DevOps**  
    - Docker / Kubernetes  
    - Flask  /  Streamlit 
    - CI/CD Pipelines  
    """)

    st.subheader("📈 Skill Proficiency")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Core ML", "90%", "↑")
    col_b.metric("Data Science & AI", "85%", "↑")
    col_c.metric("Deployment", "80%", "→")

# Education Tab
with tab5:
    st.header("📚 Education")
    st.markdown("""
    **B.Tech in Electronics & Communication Engineering**  
    Siddhartha Institute of Technology, Hyderabad  
    - 🎓 Graduation Year: 2023  
    - 📊 CGPA: 6.2/10  

    **Key Courses:**  
    - Advanced Machine Learning  
    - Deep Learning & NLP  
    - Cloud for AI  
    """)

# Certifications Tab
with tab6:
    st.header("📜 Certifications & Badges")
    
    st.markdown("""
    - 🏅 **Advanced Python Programming – [Issuing Organization]** (2025)  
    - 🏅 **AI/ML Professional – [Issuing Organization]** (2025)  
    - 🏅 **Deep Learning Specialist – [Issuing Organization]** (2025)  
    - 🏅 **Neural Networks Mastery – [Issuing Organization]** (2024)  
    - 🏅 **MLOps Fundamentals – Udacity** (2024)  
    """)

    st.markdown("### 🎖️ Badges")
    st.image("https://img.shields.io/badge/Python-Certified-green")
    st.image("https://img.shields.io/badge/AI%2FML-Professional-red")
    st.image("https://img.shields.io/badge/DeepLearning-Specialist-blue")
    st.markdown("*Note: Replace bracketed placeholders with actual issuing organization names.*")


# Projects Tab
with tab7:
    st.header("📊 Featured Projects")

    with st.container():
        st.subheader("🧠 Breast Cancer Classification")
        st.markdown("A Random Forest model achieving **97% accuracy** in classifying breast cancer. Optimized for precision and recall.")
       # st.image("breast_cancer_heatmap.png", caption="Confusion Matrix Visualizing Model Performance")
        if st.button("View Code", key="breast_cancer"):
           st.write("[GitHub Repository](https://github.com/your_username/breast_cancer_model)")


    with st.container():
        st.subheader("🏠 Airbnb Rent Prediction")
        st.markdown("A regression model predicting Airbnb rental prices. Compared predicted vs. actual rent to evaluate model accuracy and identify areas for improvement.")

        if st.button("See Predictions", key="airbnb"):
           st.write("[GitHub Repository](https://github.com/your_username/airbnb_rent_prediction)")


    with st.container():
        st.subheader("🛒 Swiggy Customer Segmentation")
        st.markdown("""
        - 🔍 **Performed customer segmentation** using K-Means and Hierarchical Clustering to uncover distinct buyer personas and behavioral patterns  
        - 🛒 **Conducted Market Basket Analysis** using Apriori and FP-Growth algorithms to discover product affinity and frequent itemsets  
        - 📊 **Analyzed feature correlations** to identify key drivers of customer behavior, such as purchase frequency, product category, and time-of-day trends  
        - 📈 **Enhanced business strategy** by identifying actionable insights that improved customer targeting and personalized marketing campaigns """)
       # st.image("swiggy_corr_heatmap.png", caption="Feature Correlation Heatmap Highlighting Key Relationships")
        
    
    with st.container():
        st.subheader("📈 Sales Forecasting with XGBoost")
        st.markdown("""
       This project focused on predicting future quarterly sales for a retail business using historical sales data. The goal was to assist in **inventory planning** and **strategic decision-making** through accurate demand forecasting.

        - ✅ Utilized **XGBoost Regressor**, known for its high accuracy and ability to handle structured/tabular data  
        - 📊 Achieved a strong performance with a **93% R² score**, indicating the model effectively captured key sales patterns  
        - 🔧 Performed **feature engineering** to incorporate time-based trends, seasonality, and lag variables  
        - 📉 Visualized **actual vs predicted sales** using line plots and error metrics to interpret model behavior  
        - 🎯 Supported stakeholders in understanding future demand trends for optimized **supply chain planning**  
        """)

       # st.image("sales_forecast_trend.png", caption="Sales Forecast Over Time")
        
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: small;'>
    © 2025 Adejarla Anjali | Last Updated: June 2025 | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
