🍓 Berry Classifier

A simple CNN that classifies images as strawberries or blueberries, with a Gradio web UI and live retraining based on user feedback.

Setup

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the training script:
   py script.py
4. Run the app:
   py app.py
5. Open http://127.0.0.1:7860 in your browser

Usage

- Upload an image to classify it
- If the prediction is wrong, click Yes and select the correct label
- The model will retrain and save automatically


"This project uses the ⁠Blueberry Detection dataset by Zhengkun Li, used under ⁠CC BY-NC-SA 4.0."

"The Strawberry dataset was kindly provided by the StrawDI Team (see https://strawdi.github.io/)."
