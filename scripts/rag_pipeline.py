"""Поиск, few-shot, генерация и проверка ссылок на текущие источники."""
import json
import re
import copy
from time import perf_counter

# Импорт retrieval первым устанавливает режим OpenMP до загрузки PyTorch.
from vector_store import Retriever
from embedding_settings import ROOT
from llm_client import LLMClient

UNKNOWN = 'Я не знаю. В найденных документах нет ответа на этот вопрос.'
SCHEMA = {
    'type':'object', 'additionalProperties':False,
    'required':['status','answer','explanation','evidence'],
    'properties':{
        'status':{'type':'string','enum':['answered','not_found']},
        'answer':{'type':'string'}, 'explanation':{'type':'string'},
        'evidence':{'type':'array','maxItems':2,'items':{'type':'string'}},
    },
}


def passages(chunk):
    return [{'evidence_id':f"{chunk['chunk_id']}_s{i:02d}", 'text':text}
            for i,text in enumerate(re.split(r'(?<=[.!?])\s+', chunk['text'].strip()),1) if text]


def context_message(question, chunks):
    return json.dumps({'question':question, 'documents':[
        {'chunk_id':c['chunk_id'],'title':c['title'],'passages':passages(c)} for c in chunks
    ]}, ensure_ascii=False)


def schema_for_hits(hits):
    schema=copy.deepcopy(SCHEMA)
    schema['properties']['evidence']['items']['enum']=[p['evidence_id'] for c in hits for p in passages(c)]
    return schema


def validate_response(data, hits):
    if not isinstance(data, dict) or set(data) != {'status','answer','explanation','evidence'}:
        raise ValueError('Неверная структура ответа модели')
    if any(not isinstance(data[k], str) or not data[k].strip() for k in ['answer','explanation']):
        raise ValueError('Пустой ответ или обоснование')
    if not isinstance(data['evidence'], list):
        raise ValueError('Неверная структура подтверждений')
    if data['status'] == 'not_found':
        if data['evidence']:
            raise ValueError('Отказ не должен содержать подтверждения ответа')
        return {'status':'not_found','answer':UNKNOWN,
                'explanation':'Недостаточно сведений в найденных фрагментах.','sources':[]}
    if data['status'] != 'answered' or not 1 <= len(data['evidence']) <= 2:
        raise ValueError('Ответ не подтверждён источниками')
    available = {p['evidence_id']:(c,p['text']) for c in hits for p in passages(c)}
    sources = []
    for evidence_id in data['evidence']:
        if not isinstance(evidence_id, str) or evidence_id not in available:
            raise ValueError('Модель сослалась на предложение вне результатов текущего поиска')
        c,quote=available[evidence_id]
        sources.append({k:c[k] for k in ['chunk_id','source','title','start_char','end_char']} | {'quote':quote,'evidence_id':evidence_id})
    return {'status':'answered','answer':data['answer'].strip(),
            'explanation':data['explanation'].strip(),'sources':sources}


def load_examples(chunks):
    examples = json.loads((ROOT/'prompts/few_shot.json').read_text(encoding='utf-8'))
    by_id = {c['chunk_id']:c for c in chunks}
    messages = []
    for example in examples:
        chunk = by_id[example['chunk_id']]
        validate_response(example['response'], [chunk])
        messages.extend([
            {'role':'user','content':context_message(example['question'], [chunk])},
            {'role':'assistant','content':json.dumps(example['response'],ensure_ascii=False)},
        ])
    return messages


class RAGBot:
    def __init__(self):
        self.retriever = Retriever()
        self.client = LLMClient()
        self.system = (ROOT/'prompts/rag_system.txt').read_text(encoding='utf-8')
        self.examples = load_examples(self.retriever.chunks)

    def ask(self, question):
        started = perf_counter()
        retrieval = self.retriever.search(question, top_k=3)
        hits = retrieval['hits']
        if not hits:
            return {'status':'not_found','answer':UNKNOWN,'explanation':'Поиск не вернул фрагментов.',
                    'sources':[], 'question':question, 'retrieval':retrieval,
                    'total_seconds':round(perf_counter()-started,6)}
        messages = [{'role':'system','content':self.system}, *self.examples,
                    {'role':'user','content':context_message(question,hits)}]
        generation = self.client.complete(messages, schema_for_hits(hits))
        result = validate_response(generation['data'], hits)
        result.update(question=question, retrieval=retrieval,
                      generation=generation, total_seconds=round(perf_counter()-started,6))
        return result


def display_result(result):
    print('\n' + result['answer'])
    print('Обоснование: ' + result['explanation'])
    for source in result['sources']:
        print(f"[{source['chunk_id']}] {source['title']} — {source['source']} [{source['start_char']}:{source['end_char']}]")
        print('Цитата: ' + source['quote'])
    print(f"Время обработки: {result['total_seconds']:.2f} с\n")
