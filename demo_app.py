import gradio as gr
from hybrid import hybrid_answer

def answer_question(image, question):
    image.save("temp_image.jpg")
    return hybrid_answer("temp_image.jpg", question)

demo = gr.Interface(
    fn=answer_question,
    inputs=["image", "text"],
    outputs="text",
    title="Assistive VQA Demo",
    description="Upload an image and ask a question about it. The model will respond accordingly."
)

if __name__ == "__main__":
    demo.launch()
