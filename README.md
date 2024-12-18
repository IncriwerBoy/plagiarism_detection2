# # Plagiarism Detection

## Overview
This project is designed to detect plagiarism in text documents by comparing and analyzing the similarity between different pieces of text using machine learning techniques, specifically Random Forest, and XGBoost.The project is deployed using Streamlit and can be accessed via the following link: [Project Dashboard](https://plagarismdetection2.streamlit.app/).

## Features
- Text similarity measures
- Random Forest classifier for plagiarism detection
- XGBoost classifier for category recognition
- Streamlit-based user interface

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/IncriwerBoy/plagarism_detection2.git
2. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
2. Upload or paste text to check for plagiarism.

## Files
- 'app.py': Streamlit application code.
- 'helper.py': Text processing utilities.
- 'artifacts/model_category.pkl': Trained machine learning model for category.
- 'artifacts/model_class.pkl': Trained machine learning model for class.

## Contributing
Feel free to fork the project, submit issues, or create pull requests.

## License
This project is licensed under the MIT License.