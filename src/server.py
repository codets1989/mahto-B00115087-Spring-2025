from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer


app = FastAPI()

model_path = "./t5-grammar-correction"  
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)


class TextRequest(BaseModel):
    text: str

@app.post("/correct")
async def correct_grammar(request: TextRequest):
    try:
        model.eval()  
        input_text = f"grammar :{request.text}"
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        print("Decoded Input:", tokenizer.decode(inputs["input_ids"][0]))  # Debugging

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True
        )

        if outputs is None or len(outputs) == 0:
            return {"corrected_text": "No correction generated."}

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"corrected_text": corrected_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


