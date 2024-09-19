"""
请求各个语言模型
"""
import threading
import time

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]  # 使用可变对象以便在内部修改
            exception = [None]  # 用于存储异常

            def worker():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")
            if exception[0] is not None:
                raise exception[0]
            return result[0]

        return wrapper
    return decorator


def deepseek_coder_v2_lite_init():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_path = '/mnt/coai_nas/qianhu/pretrained_models/modelscope/DeepSeek-Coder-V2-Lite-Instruct/' # put your path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print('finish load deepseek_coder_v2_lite')
    return tokenizer, model

@timeout(30)
def request_deepseek_coder_v2_lite(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] >= 8000:
        return ''
    outputs = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    return response

def codegeex_4_9b_init():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_path = '/mnt/coai_nas/qianhu/pretrained_models/modelscope/codegeex4-all-9b/' # put your path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print('finish load codegeex_4_9b')
    return tokenizer, model

@timeout(30)
def request_codegeex_4_9b(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] >= 8000:
        return ''
    outputs = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 256)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def starcoder_2_7b_init():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_path = '/mnt/coai_nas/qianhu/pretrained_models/modelscope/starcoder2-7b/' # put your path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print('finish load starcoder_2_7b')
    return tokenizer, model

@timeout(30)
def request_starcoder_2_7b(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] >= 8000:
        return ''
    outputs = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 256)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def codegemma_7b_init():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_path = '/mnt/coai_nas/qianhu/pretrained_models/modelscope/codegemma-7b/' # put your path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print('finish load codegemma_7b')
    return tokenizer, model

@timeout(30)
def request_codegemma_7b(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] >= 8000:
        return ''
    outputs = model.generate(**inputs, max_length=inputs.input_ids.shape[1] + 256)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



if __name__ == '__main__':
    request_codestral_7b('123')
