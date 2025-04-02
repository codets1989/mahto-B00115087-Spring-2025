from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import DistilBertTokenizer, DistilBertForTokenClassification
import re
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
model_path = "./t5-grammar-correction"  
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

model_path1 = "./distilbert-grammar-correction"
model1 = DistilBertForTokenClassification.from_pretrained(model_path1)
tokenizer1 = DistilBertTokenizer.from_pretrained(model_path1)

class TextRequest(BaseModel):
    text: str

@app.post("/correct")
async def correct_grammar(request: TextRequest):
    try:
        model.eval()
        print("Original text:", request.text)
        sentences = re.split(r'[.!?]\s+', request.text)
        print("Split Sentences:", sentences)
        corrected_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue  

            input_text = f"grammar: {sentence}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                num_beams=5,
                num_return_sequences=1,
                early_stopping=True
            )

            if outputs is None or len(outputs) == 0:
                corrected_sentences.append(sentence)  
            else:
                corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_sentences.append(corrected_text)

        
        final_corrected_text = " ".join(corrected_sentences)

        return {"corrected_text": final_corrected_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


def classify_grammar(text: str):
    inputs = tokenizer1(text, return_tensors="pt", truncation=True)
    outputs = model1(**inputs)
    
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()  
    tokens = tokenizer1.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    incorrect_tokens = [tokens[i] for i in range(len(tokens)) if predictions[i] == 1]

    if incorrect_tokens:
        return {"status": "Incorrect", "errors": incorrect_tokens}
    else:
        return {"status": "Correct", "errors": []}

@app.post("/check_grammar")
def check_grammar(request: TextRequest):
    sentences = re.split(r'[.!?]\s+', request.text)
    results = []

    for sentence in sentences:
        result = classify_grammar(sentence)
        results.append({"sentence": sentence, "status": result["status"], "errors": result["errors"]})

    return {"results": results}