from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

# Загрузка датасета
dataset = load_dataset("IlyaGusev/ru_turbo_saiga")

# Разделение датасета
train_dataset = dataset['train']
eval_dataset = dataset['validation'] if 'validation' in dataset else None

# Загрузка токенизатора
model_name = "cointegrated/rut5-base-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Функция для токенизации данных
def preprocess_function(examples):
    inputs = [f"user: {instruction}\nassistant:" for instruction in examples["instruction"]] # Формат для диалогов
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["output"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Токенизация датасета
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

if eval_dataset:
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)
else:
    tokenized_eval_dataset = None


# Загрузка модели
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./rut5_saiga_finetuned_multitask",
    evaluation_strategy="epoch",
    learning_rate=2e-5, # Уменьшено learning rate
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3, # Кол-во эпох обучения
    weight_decay=0.01,
    logging_dir="./logs_multitask",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# Дообучение модели
trainer.train()

# Сохранение модели
trainer.save_model("./rut5_saiga_finetuned_multitask")

def generate_response(query, model, tokenizer):
    input_text = f"user: {query}\nassistant:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Пример использования (после дообучения)
query = "Как получить кэшбэк в Т-Банке?"
response = generate_response(query, model, tokenizer)
print(f"Запрос: {query}")
print(f"Ответ: {response}")
