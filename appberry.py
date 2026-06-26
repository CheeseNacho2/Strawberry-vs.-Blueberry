import gradio as gr
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from PIL import Image as PILImage

model = keras.models.load_model('berry_classifier.keras')

current_image = None

def classify(image):
    global current_image
    current_image = image

    image_resized = PILImage.fromarray(image.astype('uint8')).resize((64, 64))
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = "Strawberry" if prediction[0] > 0.5 else "Blueberry"
    
    return label, gr.Row(visible=True)

def retrain(correct_label):
    global current_image
    if current_image is None:
        return "No image to train on."

    image_resized = PILImage.fromarray(current_image.astype('uint8')).resize((64, 64))
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    label = np.array([1.0]) if correct_label == "Strawberry" else np.array([0.0])

    model.fit(img_array, label, epochs=3, verbose=0)
    model.save('berry_classifier.keras')

    return f"Model updated with correct label: {correct_label}. Thank you for your feedback.", gr.Row(visible=False)

def clear():
    global current_image
    current_image = None
    # returns: image, prediction, feedback, correction row, label row
    return None, "", "", gr.Row(visible=False), gr.Row(visible=False)

def show_correction():
    # When user clicks No — show the label selection row
    return gr.Row(visible=False), gr.Row(visible=True)

def hide_correction():
    # When user clicks Yes — hide everything, prediction was correct
    return gr.Row(visible=False), gr.Row(visible=False), "Thank you for your feedback!"

with gr.Blocks() as app:
    gr.HTML("""
        <div style="text-align: center; font-family: 'Poppins', sans-serif;">
            <h1 style="font-size: 48px; color: #e63946;">🍓 Berry Classifier</h1>
            <p style="font-size: 18px; color: #555;">Upload an image to classify it as a strawberry or blueberry.</p>
        </div>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap" rel="stylesheet">
    """)

    with gr.Row():
        image_input = gr.Image()
        prediction_output = gr.Text(label="Prediction")

    with gr.Row():
        classify_btn = gr.Button("Classify 🍓")
        clear_btn = gr.Button("Clear")

    # Yes/No row — always visible after prediction
    with gr.Row(visible=False) as yn_row:
        gr.Markdown("### Was the prediction wrong?")
        yes_btn = gr.Button("Yes")
        no_btn = gr.Button("No")

    # Correction row — hidden until user clicks Yes
    with gr.Row(visible=False) as correction_row:
        gr.Markdown("### Select the correct label:")
        strawberry_btn = gr.Button("Strawberry 🍓")
        blueberry_btn = gr.Button("Blueberry 🫐")

    feedback_output = gr.Text(label="Training Status")

    # Button logic
    classify_btn.click(fn=classify, inputs=image_input, outputs=[prediction_output, yn_row])

    yes_btn.click(fn=show_correction, inputs=None, outputs=[yn_row, correction_row])
    no_btn.click(fn=hide_correction, inputs=None, outputs=[yn_row, correction_row, feedback_output])
    image_input.clear(fn=clear, inputs=None, outputs=[image_input, prediction_output, feedback_output, yn_row, correction_row])

    strawberry_btn.click(fn=lambda: retrain("Strawberry"), inputs=None, outputs=[feedback_output, correction_row])
    blueberry_btn.click(fn=lambda: retrain("Blueberry"), inputs=None, outputs=[feedback_output, correction_row])

    clear_btn.click(fn=clear, inputs=None, outputs=[image_input, prediction_output, feedback_output, yn_row, correction_row])

app.launch()