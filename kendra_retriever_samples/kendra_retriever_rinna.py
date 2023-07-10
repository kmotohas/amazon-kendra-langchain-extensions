from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import os


def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]
    endpoint_name = os.environ["RINNA_ENDPOINT"]

    class RinnaContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps(
                {
                    "instruction": "",
                    #"input": prompt.replace("\n", "<NL>"),
                    "prompt": prompt.replace("\n", "<NL>"),
                    **model_kwargs,
                }
            )
            print("prompt: ", prompt)
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            #return response_json.replace("<NL>", "\n")
            #print(response_json)
            return response_json["result"]
        
    content_handler = RinnaContentHandler()

    llm=SagemakerEndpoint(
            endpoint_name=endpoint_name, 
            region_name=region, 
            model_kwargs={
                "temperature":1e-10,
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature" : 0.7,
                "repetition_penalty": 1.3,
            },
            content_handler=content_handler
        )

    retriever = AmazonKendraRetriever(
        index_id=kendra_index_id,
        region_name=region,
        top_k=3,
        credentials_profile_name=None,
        attribute_filter={
            "EqualsTo": {      
                "Key": "_language_code",
                "Value": {
                    "StringValue": "ja"
                    }
                }
        }
#        return_source_documents=True
    )

    prompt_template = """
システム: システムは資料から抜粋して質問に答えます。資料にない内容には答えず、正直に「わかりません」と答えます。

{context}

上記の資料に基づいて以下の質問について資料から抜粋して回答を生成します。資料にない内容には答えず「わかりません」と答えます。
ユーザー: {question}
システム:
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

def run_chain(chain, prompt: str, history=[]):
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }

if __name__ == "__main__":
    chain = build_chain()
    result = run_chain(chain, "What's SageMaker?")
    print(result['answer'])
    if 'source_documents' in result:
        print('Sources:')
        for d in result['source_documents']:
            print(d.metadata['source'])
