import os
os.environ["OPENAI_API_BASE"] = ''
os.environ["OPENAI_API_KEY"] = ''

from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

from langchain.output_parsers import CommaSeparatedListOutputParser
 
output_parser = CommaSeparatedListOutputParser()


import re
def delelem(data):
    res_del = []
    for i in data:
        # 使用空字符替换掉间隔符
        a = re.sub(r'\s', '', i)
        # 使用精准匹配，匹配连续出现的符号;并用空字符替换他
        b = re.sub(r'\W{2,}', '', a)
        # 使用空字符替换空格
        c = re.sub(r' ', '', b)
        # 去除回车符号
        a = re.sub(r'\n', '', i)
        res_del.append(c)
    ans = ""
    for i in res_del:
        ans+=i
    return ans



# 获取关键词
def getKeywordsList(sentence):
    llm = OpenAI(temperature=.0)
    template = """任务：名词抽取，提取以下句子中的特殊名词：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    extract_nouns_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="keywords")

    llm = OpenAI(temperature=.3)
    template = """你已经从句子中抽取了，特殊名词了，很棒，但现在名词可能有些太多了。任务：挑选下列名词中，最难以理解的特殊名词，不超过五个，并且按一定的格式输出。以下是一些名词：{keywords}"""
    prompt_template = PromptTemplate(input_variables=["keywords"], template=template)
    special_nouns_extractor_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="keywords_1")

    llm = OpenAI(temperature=.3)
    from langchain.output_parsers import CommaSeparatedListOutputParser
    output_parser = CommaSeparatedListOutputParser()
    ans = extract_nouns_chain.run(sentence)
    ans = special_nouns_extractor_chain.run(ans)

    template = """你已经获取了一些关键词，很不错，但是我想要按一定格式输出，请你用逗号分开以下词语。任务：梳理格式。要求：将句子中的词用逗号隔开，去除特殊符号：""" + ans
    prompt = PromptTemplate(template=template, input_variables=[], output_parser=output_parser)
    keyword_separator_chain = LLMChain(prompt=prompt, llm=llm)
    ans = keyword_separator_chain.predict()
    
    ans = delelem(ans)
    ans = ans.split(',')
    return ans

# 解释关键词
def explainKeywordsList(keywords):
    llm = OpenAI(temperature=.1)
    template = """解释名词：我对这个名词不太了解，请你帮我简单解释一下，帮助我理解，五十字以内不知道的时候就说"不知道"。
    在>>> 和 <<<之间是需要解释的词语。
    Extracted:<anser or "不知道">
    >>> {keyword} <<<
    Extracted:"""
    prompt_template = PromptTemplate(input_variables=["keyword"], template=template)
    explain_nouns_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="keyword_explaination")
    ans_list = []
    for keyword in keywords:
        ans = explain_nouns_chain.run(keyword)
        ans = delelem(ans)
        ans_list.append(ans)
    return ans_list





from aiohttp.client import request
from tempfile import template
from langchain.chains import LLMChain
from langchain.chains import LLMRequestsChain
from langchain.prompts import PromptTemplate


def research(request_chain,question):
    inputs={
        "query":question,
        # "url":"https://www.baidu.com/s?wd="+question.replace(" ","+")
        "url":"https://cn.bing.com/search?q="+question.replace(" ","+")
    }
    # 运行一下就会通过openAI提取搜索结果
    ans = request_chain(inputs)
    print(ans)
    return ans['output']

import time
def research_list(questions):
    # 定义搜索模板
    template = '''在>>> 和 <<<直接是来自搜索引擎的原始搜索结果。
    我想要了解{query}的信息，请把关于'{query}'的信息从里面提取出来，如果里面没有相关信息的话就说“找不到”，
    请使用以下格式：
    Extracted:<anser or "找不到">
    >>> {requests_result} <<<
    Extracted:
    '''

    PROMPT = PromptTemplate(
        input_variables=["query","requests_result"],
        template=template,
    )
    request_chain = LLMRequestsChain(text_length=3300,llm_chain=LLMChain(llm = OpenAI(temperature = 0.3),prompt=PROMPT))
    ans_list = []
    for question in questions:
        ans = research(request_chain,question)
        ans_list.append(ans)
        time.sleep(20)
    return ans_list



def explain_sentence(sentence):
    llm = OpenAI(temperature=.0)
    template = """这是一个专利的摘要，请你解释这段摘要，以便我理解，只需要说出结论，五十字左右：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    explain_sentence_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="sentence_explain")
    ans = explain_sentence_chain.run(sentence)
    ans = delelem(ans)
    ans = ans.split(',')
    return ans

def explain_sentence_2(sentence):
    llm = OpenAI(temperature=.0)
    template = """这是一个专利的摘要，请你解释这段摘要，说明其做了什么，以便我理解，只需要说出结论，五十字左右：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    explain_sentence_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="sentence_explain")
    ans = explain_sentence_chain.run(sentence)
    ans = delelem(ans)
    ans = ans.split(',')
    return ans


def pre_value(sentence):
    llm = OpenAI(temperature=.3)
    template = """专利价值评估任务：这是一个专利的摘要，请你评价这段摘要，以便我评价其价值：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    pre_value_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="sentence_explain")
    pre_value_chain.run(sentence)
    ans = delelem(ans)
    ans = ans.split(',')
    return ans

def pre_value_conclusion(sentence):
    llm = OpenAI(temperature=.3)
    template = """专利价值评估任务：这是一个专利的摘要，请你评价这段摘要，以便我评价其价值，只需要说出结论。摘要：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    pre_value_chain_conclusion = LLMChain(llm=llm, prompt=prompt_template, output_key="sentence_explain")
    ans = pre_value_chain_conclusion.run(sentence)
    ans = delelem(ans)
    ans = ans.split(',')
    return ans






def ans_abstract_500(sentence):
    llm = OpenAI(temperature=.0)
    template = """缩写句子，要求：务必保留原本意思，不要缩写太多。{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    abstract_chain = LLMChain(llm=llm, prompt=prompt_template)
    ans = abstract_chain.run(sentence)
    ans = delelem(ans)
    return ans

def ans_abstract(sentence):
    llm = OpenAI(temperature=.0)
    template = """缩写句子到六十字：帮我缩写到六十字以内：{sentence}"""
    prompt_template = PromptTemplate(input_variables=["sentence"], template=template)
    abstract_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="keyword_explaination")
    ans = abstract_chain.run(sentence)
    ans = delelem(ans)
    return ans

def ansList_abstract(sentence_list):
    for index,_ in enumerate(sentence_list):
        counter = 0
        while len(sentence_list[index]) > 60 and counter < 2:
            sentence_list[index] = ans_abstract(sentence_list[index])
            counter+=1
        if len(sentence_list[index]) > 60:
            sentence_list[index] = sentence_list[index][0:60]
    return sentence_list





def all_in_all(sentence,isSearchOnInternet = False):
    keywords_list = getKeywordsList(sentence)

    keywords_explaination_list = explainKeywordsList(keywords_list)
    
        

    sentence_explaination = explain_sentence(sentence)
    pre_valuation = pre_value(sentence)
    pre_value_conclusion_ = pre_value_conclusion(sentence)

    keywords_search_rsp_list = []
    if isSearchOnInternet:
        keywords_search_rsp_list = research_list(keywords_list)

    # 返回： 关键词序列 ， 关键词解释 ， 句子解释 ， 专利评价 ， 专利评价的结论 ， 搜索结果
    return keywords_list,keywords_explaination_list,sentence_explaination,pre_valuation,pre_value_conclusion_,keywords_search_rsp_list


def knowledge_enhance(sentence,isSearchOnInternet = False):
    sentence_explaination = ''



    keywords_list = getKeywordsList(sentence)

    keywords_explaination_list = explainKeywordsList(keywords_list)
    keywords_explaination_list = ansList_abstract(keywords_explaination_list)

    keywords_search_rsp_list = []

    if isSearchOnInternet:
        keywords_search_rsp_list = research_list(keywords_list)
        keywords_search_rsp_list = ansList_abstract(keywords_search_rsp_list)

    # sentence_explaination = explain_sentence(sentence)
    # sentence_explaination = ans_abstract(sentence_explaination)

    sentence_explaination_2 = explain_sentence_2(sentence)

    return {"keywords_list":keywords_list,
            "keywords_explaination_list":keywords_explaination_list,
            "keywords_search_rsp_list":keywords_search_rsp_list,
            "sentence_explaination":sentence_explaination,
            "sentence_explaination_2":sentence_explaination_2}




def combine(inputs,sentence):
    ans = ""
    # ans = ans + "原文：" + sentence
    ans = ans + sentence
    knowledge = "相关知识："
    knowledge += inputs['sentence_explaination_2'][0]
    knowledge += "关键词解析："
    for kw,kwe in zip(inputs['keywords_list'],inputs['keywords_explaination_list']):
        knowledge += kw + ':' + kwe
    counter = 0
    while len(ans) + len(knowledge) > 510 and counter < 3:
        counter+=1
        knowledge = ans_abstract_500(knowledge)
    else:
        knowledge = knowledge[:510]
    return ans + knowledge


def get_knowledge_enhance(st):
    sentence = {"sentence" :st}
    ans = knowledge_enhance(sentence,isSearchOnInternet=False)
    ans = combine(ans,st)
    return ans

# print("======================")
# st = "一种全谷物营养粉及其制备方法，全谷物营养粉成分包括原料和辅料，原料包括：3麸皮、糙米、甜玉米、小米、红豆、小麦、大豆、糯米、高粱、薏米、荞麦、燕麦；辅料包括：白砂糖、低聚果糖、麦芽精糊、黑芝麻粉。该营养粉制备方法包括：将原料组分去除杂质，碾磨成粉末，用过滤网筛出粉粒，再进行碾磨，以此重复8次后，得全谷粉；取辅料研磨成超微粉，并加入全谷粉搅拌均匀，得混合粉，加水进行磨浆，过筛，变成全谷物浆，真空干燥，得全谷物粉，低温冷藏，微波灭菌，真空包装即得。该全谷物营养粉能补充营养，调节身体机能，对缺乏血色的女性能提供好肤色，且有促进消化、促进营养吸收及提高免疫力的作用。"

# ans = get_knowledge_enhance(st)
# print(ans)




import csv

# 定义原始CSV文件和新CSV文件的路径
original_csv_file = './data/test.csv'
new_csv_file = './data/test_llm.csv'

# 打开原始CSV文件进行读取，并创建新CSV文件进行写入
with open(original_csv_file, 'r', encoding='utf-8') as original_file, \
     open(new_csv_file, 'w', newline='', encoding='utf-8') as new_file:

    # 创建CSV读写对象
    csv_reader = csv.reader(original_file)
    csv_writer = csv.writer(new_file)

    # 写入标题行
    header = next(csv_reader)  # 读取原始CSV文件的标题行
    header.append('new_column')  # 添加新的列标题
    csv_writer.writerow(header)  # 写入新CSV文件的标题行

    # 逐行读取原始CSV文件中的数据，并添加新列数据，然后写入新CSV文件
    for row in csv_reader:
        # 添加新的列数据
        
        ans = get_knowledge_enhance(row[0])
        row.append(ans)

        # 写入新CSV文件
        csv_writer.writerow(row)
        break
        
print("New CSV file with additional column has been created successfully.")