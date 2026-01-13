# Heart Disease Prediction App

This project predicts the likelihood of heart disease using a Machine Learning model deployed with Streamlit.

## Files

- `Model.ipynb`: Original Jupyter Notebook with data analysis and model training.
- `train_model.py`: Python script to train the model and save it as `heart_disease_model.sav`.
- `heart_disease_model.sav`: The trained logistic regression model.
- `app.py`: The Streamlit web application.
- `requirements.txt`: List of dependencies required to run the app.
- `heart.csv`: The dataset used for training.

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## How to Deploy (Get a hosted link)

The easiest way to host this for free is using **Streamlit Community Cloud**.

1. **Upload this code to GitHub:**
   - Create a new repository on GitHub.
   - Upload all the files in this folder (`app.py`, `heart_disease_model.sav`, `requirements.txt`, etc.) to the repository.

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Sign up/Login with your GitHub account.
   - Click "New app".
   - Select your repository, branch (usually `main`), and the main file path (`app.py`).
   - Click "Deploy".

3. **Get your link:**
   - Once deployed, you will get a URL (e.g., `https://your-app-name.streamlit.app`) which you can put on your resume.
