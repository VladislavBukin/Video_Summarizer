import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def generate_summary(text):
    inputs = tokenizer.encode("сжать:"+text,return_tensors="pt",max_length=1024,truncation = True)
    summary_ids = model.generate(inputs,max_length=int(len(text)/4), min_length = int(len(text)/8), length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return summary

# Пример текста для суммаризации
#text_to_summarize = """
#В 2001 году Лукашенко утвердил новую Конституцию. 
#По новой Конституции, президенту разрешено быть избранным 
#не более чем трижды подряд. В 2004 году Лукашенко был переизбран 
#на четвёртый срок. По конституции, он не имел права баллотироваться 
#на пятый срок. 
#"""
text_to_summarize = """
The world is facing a climate crisis unlike anything we've seen before. Rising temperatures, melting ice caps, and extreme weather events are becoming more frequent. It's clear that urgent action is needed to address this global challenge. Governments, businesses, and individuals must work together to reduce carbon emissions, invest in renewable energy sources, and protect our planet for future generations.

In the field of artificial intelligence, advancements are happening at a rapid pace. From self-driving cars to smart assistants, AI is transforming industries and changing the way we live and work. However, with these advancements come ethical considerations. Questions about data privacy, algorithm bias, and the impact on jobs are being raised. As AI continues to evolve, it's crucial to have discussions and policies in place to ensure responsible and ethical development.

The COVID-19 pandemic has had a profound impact on societies worldwide. Lockdowns, social distancing, and travel restrictions have become the new norm. While vaccines offer hope for a return to normalcy, the pandemic has highlighted the importance of global cooperation in tackling health crises. Lessons learned from this experience will shape how we prepare for and respond to future pandemics.

Space exploration continues to capture the imagination of people around the world. From Mars rovers to plans for lunar missions, humanity's reach into the cosmos is expanding. Discoveries about distant planets, black holes, and the origins of the universe are pushing the boundaries of our understanding. As we look to the stars, we are reminded of the vastness of the universe and our place within it.
"""
text_to_summarize2 = """
Мир сталкивается с климатическим кризисом, подобного которому мы еще не видели. Повышение температуры, таяние ледяных шапок и экстремальные погодные явления становятся все более частыми. Очевидно, что для решения этой глобальной проблемы необходимы срочные действия. Правительства, предприятия и частные лица должны работать сообща, чтобы сократить выбросы углекислого газа, инвестировать в возобновляемые источники энергии и защитить нашу планету для будущих поколений.

В области искусственного интеллекта достижения происходят быстрыми темпами. От самоуправляемых автомобилей до умных помощников ИИ трансформирует отрасли и меняет наш образ жизни и работы. Однако вместе с этими достижениями возникают этические соображения. Возникают вопросы о конфиденциальности данных, предвзятости алгоритмов и влиянии на рабочие места. Поскольку искусственный интеллект продолжает развиваться, крайне важно проводить обсуждения и разрабатывать политику для обеспечения ответственного и этичного развития.

Пандемия COVID-19 оказала глубокое воздействие на общество во всем мире. Карантин, социальное дистанцирование и ограничения на поездки стали новой нормой. Хотя вакцины дают надежду на возвращение к нормальной жизни, пандемия подчеркнула важность глобального сотрудничества в преодолении кризисов в области здравоохранения. Уроки, извлеченные из этого опыта, будут определять то, как мы будем готовиться к будущим пандемиям и реагировать на них.

Освоение космоса продолжает захватывать воображение людей по всему миру. Начиная с марсоходов и заканчивая планами полетов на Луну, возможности человечества в освоении космоса расширяются. Открытия о далеких планетах, черных дырах и происхождении Вселенной раздвигают границы нашего понимания. Когда мы смотрим на звезды, они напоминают нам о необъятности Вселенной и нашем месте в ней.
"""

# Вызов функции для суммаризации текста
result_summary = generate_summary(text_to_summarize)
result_summary2 = generate_summary(text_to_summarize2)

print("Оригинальный текст:")
print(text_to_summarize)

print("\nСгенерированный реферат:")
print(result_summary)

print("Оригинальный текст:")
print(text_to_summarize2)

print("\nСгенерированный реферат:")
print(result_summary2)
