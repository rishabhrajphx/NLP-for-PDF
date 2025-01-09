from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned-model")
model = AutoModelForTokenClassification.from_pretrained("./models/fine-tuned-model")

def predict_entities(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2).tolist()
    entities = []
    for i, pred in enumerate(predictions[0]):
        if pred != 0:
            token = tokenizer.decode([inputs.input_ids[0][i]])
            entities.append((token, model.config.id2label[pred]))
    return entities

if __name__ == "__main__":
    text = "John Doe 123 Main St. (555) 123-4567 john.doe@example.com github.com/johndoe linkedin.com/in/johndoe University of Example Computer Science"
    print(predict_entities(text))