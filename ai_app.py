import os, json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import load_prompt
from langchain.llms import OpenAI
from langchain import LLMChain

def choose_fields_chain(description, rows):
    prompt = load_prompt(os.path.abspath("prompts/choose_fields.json"))
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    result = llm_chain.run({"description": description, "rows": rows})
    return result

def select_bin_method_chain(methods):
    prompt = load_prompt(os.path.abspath("prompts/select_bin.json"))
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    result = llm_chain.run({"method": methods})
    return result

def apply_method_chain(dataset, method_code):
    prompt = load_prompt(os.path.abspath("prompts/apply_method.json"))
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    result = llm_chain.run({"dataset": dataset, "method_code": method_code})
    return result

def plot_histogram_chain(df):
    prompt = load_prompt(os.path.abspath("prompts/plot_histogram.json"))
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI())
    result = llm_chain.run({"dataset": json.dumps(df.to_dict(orient='records'))})
    return result

if __name__ == '__main__':
    # read csv file
    df = pd.read_csv(os.path.abspath('abalone.csv'))
    # get the first 10 rows
    df = df.head(10)
    # show columns
    print(df.columns)
    # choose fields for histogram
    fields = choose_fields_chain("This is a dataset of abalone", json.dumps(df.to_dict(orient='records')))
    fields = json.loads(fields)
    print(fields)
    # generate code for bin choosing methods
    for method in ["Sturges’s Rule", ", Scott’s normal reference rule", "Cross validation"]:
        method_code = select_bin_method_chain(method)
        print("code method:")
        print(method_code)
        # apply method
        bins = apply_method_chain(dataset=json.dumps(df.to_dict(orient='records')), method_code=method_code)
        print("applying method:")
        print(bins)
        plot_code = plot_histogram_chain(df)
        print("plot code:")
        print(plot_code)

    
    


