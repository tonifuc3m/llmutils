"""
Chat wrappers for models not in LLM tools

Usage example:
```python
# From https://python.langchain.com/v0.2/docs/integrations/chat/llama2_chat/
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

from langchain_community.llms import HuggingFaceTextGenInference

llm = HuggingFaceTextGenInference(
    inference_server_url="http://127.0.0.1:8080/",
    max_new_tokens=512,
    top_k=50,
    temperature=0.1,
    repetition_penalty=1.03,
)

model = Llama2Chat(llm=llm) # <------------------------------------------------
chain = LLMChain(llm=model, prompt=prompt_template) # <------------------------
chain.run(
        text="What can I see in Vienna? Propose a few locations. Names only, no details."
    )
```
"""


from langchain_experimental.chat_models.llm_wrapper import ChatWrapper

class Llama3Chat(ChatWrapper):
    """Wrapper for Llama-3-instruct model.
    Source: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
    """
    # TODO: test if there are issues. LLama-3-instruct has problems with the
    # EOS token. It should be <|end_of_text|>, but in chat mode it is <|eot_id|>
    # So depending on the instruct fine-tuning tokenizer_config.json, it stop 
    # generating at <|end_of_text|> or <|eot_id|>. This is a problem because if 
    # it does not stop at <|eot_id|>, the chat will continue generating 
    # indefinitely. 
    @property
    def _llm_type(self) -> str:
        return "llama-3-instruct"

    sys_beg: str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    sys_end: str = "<|eot_id|>"
    ai_n_beg: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ai_n_end: str = "<|eot_id|>"
    usr_n_beg: str = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    usr_n_end: str = "<|eot_id|>"
    usr_0_beg: str = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    usr_0_end: str = "<|eot_id|>"

class Qwen2Instruct(ChatWrapper):
    """Wrapper for Qwen-2-instruct model.
    Source: 
    infered from tokenizer.apply_chat_template() and 
    https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/blob/main/tokenizer_config.json
    """
    # TODO: test if there are issues.
    @property
    def _llm_type(self) -> str:
        return "qwen-2-instruct"

    sys_beg: str = "<|im_start|>system\n"
    sys_end: str = "<|im_end|>\n"
    ai_n_beg: str = "<|im_start|>assistant\n"
    ai_n_end: str = "<|im_end|>\n"
    usr_n_beg: str = "<|im_start|>user\n"
    usr_n_end: str = "<|im_end|>\n"
    usr_0_beg: str = "<|im_start|>user\n"
    usr_0_end: str = "<|im_end|>\n"