import jsonlines
import warnings
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch


# Шаг 1: Извлечение данных из JSONL файла
def extract_data_from_jsonl(file_path):
    urls = []
    texts = []
    titles = []
    summaries = []
    dates = []

    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            urls.append(item['url'])
            texts.append(item['text'])
            titles.append(item['title'])
            summaries.append(item['summary'])
            dates.append(item['date'])

    return urls, texts, titles, summaries, dates


# Шаг 2: Подготовка данных для дообучения
class CustomDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_input_length=512, max_target_length=64):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(text, max_length=self.max_input_length, padding="max_length", truncation=True,
                                return_tensors="pt")
        targets = self.tokenizer(summary, max_length=self.max_target_length, padding="max_length", truncation=True,
                                 return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": targets["input_ids"].flatten()
        }


if __name__ == "__main__":
    # Параметры файлов
    train_file_path = './datasets/gazeta_train.jsonl'
    test_file_path = './datasets/gazeta_test.jsonl'
    save_path = "./models/finetuned_bart"

    # Шаг 1: Извлечение данных
    urls, texts, titles, summaries, dates = extract_data_from_jsonl(train_file_path)

    # Шаг 2: Подготовка данных для дообучения
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    train_dataset = CustomDataset(texts, summaries, tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Перенос модели и данных на CUDA (GPU)
    model_config = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').config
    model_config.num_labels = tokenizer.vocab_size
    new_model = BartForConditionalGeneration(config=model_config).to(device)

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir="./logs",
        logging_steps=100,
    )

    trainer = Trainer(
        model=new_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    # Шаг 4: Сохранение дообученной модели
    trainer.save_model(save_path)

    # Шаг 5: Оценка производительности модели на тестовой выборке
    test_urls, test_texts, test_titles, test_summaries, test_dates = extract_data_from_jsonl(test_file_path)
    test_dataset = CustomDataset(test_texts, test_summaries, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    evaluation_args = TrainingArguments(
        per_device_eval_batch_size=4,
        output_dir="./results",
    )

    trainer = Trainer(
        model=BartForConditionalGeneration.from_pretrained(save_path).to(device),
        args=evaluation_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    results = trainer.predict(test_loader)
    print("Evaluation Results:", results)
