from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
dataset = load_dataset("jfleg")
def preprocess_function(examples):
    inputs = [f"grammar: {text}" for text in examples["sentence"]]
    targets = [corrections[0] for corrections in examples["corrections"]]  
    return {"input_text": inputs, "target_text": targets}

tokenized_dataset = dataset.map(preprocess_function, batched=True)


model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]  

    return model_inputs



tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="./writing-coach-t5",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["validation"],  
    eval_dataset=tokenized_dataset["test"],         
    tokenizer=tokenizer
)
trainer.train()
model.save_pretrained("./t5-grammar-correction")
tokenizer.save_pretrained("./t5-grammar-correction")
results = trainer.evaluate()
print(results)
model_path = "./t5-grammar-correction"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
def correct_grammar(text):
    try:
        model.eval()  
        input_text = "grammar: I has an apple"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        print("Decoded Input:", tokenizer.decode(inputs["input_ids"][0]))  # Debug

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,  
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True
        )
        
        if outputs is None or len(outputs) == 0:
            return "No correction generated."

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text

    except Exception as e:
        print(f"Error: {e}")
        return text  
print(correct_grammar("I has a apple."))     