import os
os.environ['PYTHONUNBUFFERED'] = "1"
import argparse
import tqdm
import json
import pandas
import re
import pathlib
import shutil
import multiprocessing
import random
random.seed(1234)
from request_model import \
    deepseek_coder_v2_lite_init, request_deepseek_coder_v2_lite, \
    codegeex_4_9b_init, request_codegeex_4_9b, \
    starcoder_2_7b_init, request_starcoder_2_7b, \
    codegemma_7b_init, request_codegemma_7b
from prepare import filter_method_dict, run_unit_test_command
import threading
import time


MAIN_DIR = '/mnt/coai_nas/qianhu/github/Codev-Bench/'
SUBTYPE_2_MAINTYPE = {
    'METHOD': '函数',
    'IF': '判断逻辑块',
    'FOR': '循环逻辑块',
    'WHILE': '循环逻辑块',
    'TRY': '异常逻辑块',
    'CATCH': '异常逻辑块',
    'STATEMENT': '普通语句',
}
N_PROCESS = 4


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


def preview_test_llms(
        model_name='gpt_4o_mini',
        mode='prefix_suffix_full_complete_current_block_no_evidence'):
    """
    用llms进行预测
    """
    main_dir = MAIN_DIR
    source_code_dir = os.path.join(main_dir, 'Source_Code')
    copyed_source_code_dir = os.path.join(main_dir, 'Source_Code_Copy')
    cosine_path = os.path.join(main_dir, 'metadatas/test_file_cosine.output.csv')
    meta_dir = os.path.join(main_dir, 'metadatas/')
    template_path = os.path.join(main_dir, 'src/templates/llm_template.py')
    predicts_dir = os.path.join(main_dir, 'predicts', mode)
    predicts_path = os.path.join(predicts_dir, 'predictions', f'{model_name}.jsonl')
    pathlib.Path(predicts_path).parent.mkdir(parents=True, exist_ok=True)
    test_path = os.path.join(main_dir, 'prompts', f'{mode}.jsonl')
    template = open(template_path, 'r').read()
    block_dict, type_dict = {}, {}
    with open(test_path, 'r') as fo:
        for line in tqdm.tqdm(fo.readlines(), desc='read'):
            data = json.loads(line)
            block_dict[data['block_key']] = data
            if data['block_type'] not in type_dict:
                type_dict[data['block_type']] = []
            type_dict[data['block_type']].append(data['block_key'])
    block_keys = []
    for btype in ['STATEMENT', 'IF', 'FOR', 'TRY', 'METHOD']:
        if btype not in type_dict:
            continue
        random.shuffle(type_dict[btype])
        for i in range(len(type_dict[btype])):
            block_key = type_dict[btype][i]
            block_keys.append(block_key)
    
    # 批量请求结果
    def _request(procid, n_proc, block_keys):
        if model_name == 'deepseek_coder_v2_lite':
            tokenizer, model = deepseek_coder_v2_lite_init()
        elif model_name == 'codegeex_4_9b':
            tokenizer, model = codegeex_4_9b_init()
        elif model_name == 'starcoder_2_7b':
            tokenizer, model = starcoder_2_7b_init()
        elif model_name == 'codegemma_7b':
            tokenizer, model = codegemma_7b_init()
        else:
            tokenizer, model = None, None
        with open(f"{predicts_path}.{procid}", 'w') as fw:
            for bid, block_key in enumerate(tqdm.tqdm(
                    block_keys, desc=f'request {procid}/{n_proc}', position=procid)):
                if bid % n_proc != procid:
                    continue
                block_info = block_dict[block_key]
                
                # 获取prompt
                if model_name in ['gpt_4o_mini', 'claude_35_sonnet', 
                        'qwen_plus', 'qwen_2_72b', 'qwen_2_54b_moe',
                        'llama_31_405b', 'llama_31_70b', 'llama_31_8b',
                        'deepseek_v2', 'mistral_123b', 'yi_15_34b', 
                        'glm_3_6b']:
                    question = block_info['prompt'].split('<fim_middle>')[0]
                    query = template.replace('[question]', question)
                elif model_name in ['lingma_completion_v75', 
                        'lingma_completion_v76', 'codeqwen_15']:
                    query = block_info['prompt'].split('<fim_middle>')[0] + '<fim_middle>'
                elif model_name in ['deepseek_coder_v2_lite']:
                    query = block_info['prompt'].split('<fim_middle>')[0] + '<fim_middle>'
                    query = '<fim_prefix>' + query.split('<fim_prefix>')[1]
                    query = query.replace('<fim_prefix>', '<｜fim▁begin｜>').replace(
                        '<fim_suffix>', '<｜fim▁hole｜>').replace(
                        '<fim_middle>', '<｜fim▁end｜>')
                elif model_name in ['codegeex_4_9b']:
                    query = block_info['prompt'].split('<fim_middle>')[0] + '<fim_middle>'
                    query = '<fim_prefix>' + query.split('<fim_prefix>')[1]
                    query = '<|user|>\n' + query.replace(
                        '<fim_prefix>', '<|code_prefix|>').replace(
                        '<fim_suffix>', '<|code_suffix|>').replace(
                        '<fim_middle>', '<|code_middle|>') + '<|assistant|>'
                elif model_name in ['starcoder_2_7b']:
                    query = block_info['prompt'].split('<fim_middle>')[0] + '<fim_middle>'
                    query = '<fim_prefix>' + query.split('<fim_prefix>')[1]
                elif model_name in ['codegemma_7b']:
                    query = block_info['prompt'].split('<fim_middle>')[0] + '<fim_middle>'
                    query = '<fim_prefix>' + query.split('<fim_prefix>')[1]
                    query = query.replace(
                        '<fim_prefix>', '<|fim_prefix|>').replace(
                        '<fim_suffix>', '<|fim_suffix|>').replace(
                        '<fim_middle>', '<|fim_middle|>')
                
                # 请求模型
                st_time = time.time()
                # try:
                if True:
                    if model_name == 'gpt_4o_mini':
                        response = request_gpt4o(query)
                    elif model_name == 'claude_35_sonnet':
                        response = request_claude35(query)
                    elif model_name == 'qwen_plus':
                        response = request_qwen_plus(query)
                    elif model_name == 'qwen_2_72b':
                        response = request_qwen_2_72b(query)
                    elif model_name == 'qwen_2_54b_moe':
                        response = request_qwen_2_54b_moe(query)
                    elif model_name in ['lingma_completion_v75', 'lingma_completion_v76']:
                        response = request_lingma_completion(query)
                    elif model_name == 'codeqwen_15':
                        response = request_codeqwen_15(query)
                    elif model_name == 'llama_31_405b':
                        response = request_llama_31_405b(query)
                    elif model_name == 'llama_31_70b':
                        response = request_llama_31_70b(query)
                    elif model_name == 'deepseek_v2':
                        response = request_deepseek_v2(query)
                    elif model_name == 'mistral_123b':
                        response = request_mistral_123b(query)
                    elif model_name == 'yi_15_34b':
                        response = request_yi_15_34b(query)
                    elif model_name == 'glm_3_6b':
                        response = request_glm_3_6b(query)
                    elif model_name == 'deepseek_coder_v2_lite':
                        response = request_deepseek_coder_v2_lite(query, tokenizer, model)
                    elif model_name == 'codegeex_4_9b':
                        response = request_codegeex_4_9b(query, tokenizer, model)
                    elif model_name == 'starcoder_2_7b':
                        response = request_starcoder_2_7b(query, tokenizer, model)
                    elif model_name == 'codegemma_7b':
                        response = request_codegemma_7b(query, tokenizer, model)
                # except:
                #     response = ''
                et_time = time.time()
                
                # 解析结果
                if model_name in ['gpt_4o_mini', 'claude_35_sonnet', 
                        'qwen_plus', 'qwen_2_72b', 'qwen_2_54b_moe',
                        'llama_31_405b', 'llama_31_70b', 'llama_31_8b',
                        'deepseek_v2', 'mistral_123b', 'yi_15_34b',
                        'glm_3_6b']:
                    results = re.findall(r'```.*?<br>(.*?)<br>```', 
                        response.replace('\n', '<br>'))
                    if len(results) > 0:
                        predict_middle = results[-1].replace('<br>', '\n')
                    else:
                        predict_middle = ''
                elif model_name in ['lingma_completion_v75', 'codeqwen_15']:
                    predict_middle = response.split('<|endoftext|>')[0]
                elif model_name in ['deepseek_coder_v2_lite', 'codegeex_4_9b',
                        'starcoder_2_7b']:
                    predict_middle = response
                elif model_name in ['codegemma_7b']:
                    predict_middle = response.split('<|file_separator|>')[0]
                block_info['response_original_text'] = response
                block_info['response'] = predict_middle
                block_info['latency'] = et_time - st_time
                
                # 存储
                fw.write(json.dumps(block_info) + '\n')
                fw.flush()
    processes = []
    n_process = 1
    for procid in range(n_process):
        p = multiprocessing.Process(target=_request, args=(procid, n_process, block_keys))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    os.system(f"cat {predicts_path}.* > {predicts_path}")
    os.system(f"rm -f {predicts_path}.*")

def judge_right_indent(data, block_dict):
    """
    判断indent是否正确
    """
    file_path = data['func_name'].split('#')[0]
    code = open(file_path, 'r').read()
    first_line = code.split('\n')[block_dict[data['block_key']]['startLine']]
    ground_indent = ''
    for char in first_line:
        if len(char.strip()) != 0:
            break
        ground_indent += char
    for line in data['response'].split('\n')[1:]:
        line_indent = ''
        for char in line:
            if len(char.strip()) != 0:
                break
            line_indent += char
        if len(line_indent) < len(ground_indent):
            return False
    return True

def judge_not_cross_line(data):
    """
    判断没有跨行
    """
    return bool(len(data['response'].split('\n')) == 1)

def judge_empty(data):
    """
    判断是否空
    """
    return bool(data['response'] == '')

def evaluate_prediction(
        model_name='gpt_4o_mini',
        mode='prefix_suffix_full_complete_current_block_no_evidence',
        check_unittest=False,
        check_indent=False,
        check_cross_line=False,
        check_empty=False):
    """
    评测预测结果
    """
    main_dir = MAIN_DIR
    predicted_path = os.path.join(main_dir, 'predicts', mode, 'predictions', f'{model_name}.jsonl')
    log_dir = os.path.join(main_dir, 'predicts', mode, 'logs')
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    result_path = os.path.join(main_dir, 'predicts', mode, 'results', f'{model_name}.jsonl')
    pathlib.Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    source_code_dir = os.path.join(main_dir, 'Source_Code')
    copyed_source_code_dir = os.path.join(main_dir, 'Source_Code_Copy')
    meta_dir = os.path.join(main_dir, 'metadatas/')
    cosine_path = os.path.join(main_dir, 'metadatas/test_file_cosine.output.csv')
    method_line_dict, method_dict, block_dict = filter_method_dict(cosine_path, source_code_dir)
    # 获取所有单测文件
    test_dict = {}
    for repo_name in os.listdir(meta_dir):
        if os.path.isdir(os.path.join(meta_dir, repo_name)):
            test_path = os.path.join(meta_dir, repo_name, 
                'test_metadata_funcname_passed_called.jsonl')
            if os.path.exists(test_path):
                with open(test_path, 'r') as fo:
                    for line in fo:
                        data = json.loads(line)
                        test_dict[data['did']] = data
    # 将需要跑单测的文件按照项目进行拆分
    repo_test_dict = {}
    with open(predicted_path, 'r') as fo:
        for lid, line in enumerate(tqdm.tqdm(fo.readlines(), desc='evaluate')):
            data = json.loads(line)
            file_path = data['func_name'].split('#')[0]
            file_path = file_path.replace(
                '/mnt/coai_nas/qianhu/github/completion_benchmark/', MAIN_DIR)
            repo_name = file_path[len(source_code_dir)+1:].split('/')[0]
            if repo_name not in repo_test_dict:
                repo_test_dict[repo_name] = []
            repo_test_dict[repo_name].append(data)
    # 初始化仓库
    source_target_paths = []
    for repo_name in os.listdir(meta_dir):
        for root, _, filenames in os.walk(os.path.join(copyed_source_code_dir, repo_name)):
            for filename in filenames:
                source_path = os.path.join(root, filename)
                target_path = source_path.replace('Source_Code_Copy', 'Source_Code')
                source_target_paths.append((source_path, target_path))
    for source_path, target_path in tqdm.tqdm(source_target_paths, desc='init repo'):
        shutil.copy(source_path, target_path)

    @timeout(120)
    def _run_one_unit_test(data, utid, test_item, file_path, log_dir):
        predicted_code = data['response']
        # 运行预测后的代码对应的单测
        prefix = data['prefix']
        suffix = data['suffix']
        predicted_file = prefix + predicted_code + suffix
        with open(file_path, 'w') as temp_fw:
            temp_fw.write(predicted_file)
        func_name = data['func_name']
        func_name = func_name.replace(
            '/mnt/coai_nas/qianhu/github/completion_benchmark/', MAIN_DIR)
        log_path = os.path.join(log_dir, f"{func_name}_{utid}.log")
        is_pass, _ = run_unit_test_command(test_item, log_path)
        # 恢复文件
        source_path = file_path.replace('Source_Code', 'Source_Code_Copy')
        target_path = file_path
        shutil.copy(source_path, target_path)
        
        return is_pass
        
    # 多进程跑单测
    def _run_one_repo_unit_test(procid, n_proc, repo_test_dict, test_dict, block_dict):
        repo_names = list(repo_test_dict.keys())
        with open(result_path + f'.{procid}', 'w') as fw:
            datas = []
            for rid, repo_name in enumerate(repo_names):
                if repo_name != 'searcharray':
                    continue
                if rid % n_proc != procid:
                    continue
                for data in repo_test_dict[repo_name]:
                    datas.append(data)
            for data in tqdm.tqdm(datas, desc=f'evaluate {procid}/{n_proc}', position=procid):
                file_path = data['func_name'].split('#')[0]
                file_path = file_path.replace(
                    '/mnt/coai_nas/qianhu/github/completion_benchmark/', MAIN_DIR)
                if check_indent:
                    is_right_indent = judge_right_indent(data, block_dict)
                else:
                    is_right_indent = True
                if check_cross_line:
                    is_not_cross_line = judge_not_cross_line(data)
                else:
                    is_not_cross_line = True
                if check_empty:
                    is_empty = judge_empty(data)
                else:
                    is_empty = True
                if check_unittest:
                    is_test_passed = True
                    for utid, unit_test_id in enumerate(data['unit_test_ids']):
                        test_item = test_dict[unit_test_id]
                        test_item['run_command'] = test_item['run_command'].replace(
                            '/mnt/coai_nas/qianhu/github/completion_benchmark/', MAIN_DIR)
                        try:
                            is_pass = _run_one_unit_test(
                                data, utid, test_item, file_path, log_dir)
                        except TimeoutError as e:
                            is_pass = False
                        if not is_pass:
                            is_test_passed = False
                            break
                else:
                    is_test_passed = True
                data['is_all_passed'] = bool(
                    is_test_passed and is_right_indent and \
                    is_not_cross_line and is_empty)
                fw.write(json.dumps(data, ensure_ascii=False) + '\n')
                fw.flush()
    n_process = N_PROCESS
    processes = []
    for procid in range(n_process):
        p = multiprocessing.Process(
            target=_run_one_repo_unit_test,
            args=(procid, n_process, repo_test_dict, test_dict, block_dict))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def print_scores(
        model_name='gpt_4o_mini',
        mode='prefix_suffix_full_complete_current_block_no_evidence'):
    """
    获取结果
    """
    main_dir = MAIN_DIR
    result_path = os.path.join(main_dir, 'predicts', mode, 'results', f'{model_name}.jsonl')
    score_path = os.path.join(main_dir, 'predicts', mode, 'scores', f'{model_name}.jsonl')
    pathlib.Path(score_path).parent.mkdir(parents=True, exist_ok=True)
    eval_dict = {}
    for procid in range(N_PROCESS):
        with open(result_path + f'.{procid}', 'r') as fo:
            for line in fo:
                data = json.loads(line)
                main_type = SUBTYPE_2_MAINTYPE[data['block_type']]
                for mtype in [main_type, 'TOTAL']:
                    if mtype not in eval_dict:
                        eval_dict[mtype] = {'n_all': 0, 'n_pass': 0}
                    if data['is_all_passed']:
                        eval_dict[mtype]['n_pass'] += 1
                    eval_dict[mtype]['n_all'] += 1
    json.dump(eval_dict, open(score_path, 'w'), indent=4)
    for main_type in sorted(eval_dict.keys()):
        n_all = eval_dict[main_type]['n_all']
        n_pass = eval_dict[main_type]['n_pass']
        eval_dict[main_type]['pass_rate'] = pass_rate = 100.0 * n_pass / n_all
        print(f"model_name={model_name}, type={main_type}, n_all={n_all}, n_pass={n_pass}, " \
            f"acc={pass_rate:.2f}%")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='')
    parser.add_argument("--model", type=str, choices=[
        'deepseek_coder_v2_lite', 'codegeex_4_9b',
        'starcoder_2_7b', 'codegemma_7b'], default='')
    parser.add_argument("--mode", type=str, choices=[
        'prefix_suffix_full_complete_current_block_no_evidence',
        'prefix_suffix_full_complete_current_block_with_evidence',
        'prefix_full_suffix_func_empty_complete_current_block_no_evidence',
        'prefix_full_suffix_empty_complete_current_block_no_evidence',
        'complete_current_header_inner_block_completion',
        'complete_current_header_empty_completion'],
        default='prefix_suffix_full_complete_current_block_no_evidence')
    parser.add_argument("--check-unittest", action='store_true')
    parser.add_argument("--check-indent", action='store_true')
    parser.add_argument("--check-cross-line", action='store_true')
    parser.add_argument("--check-empty", action='store_true')
    args = parser.parse_args()
    
    if args.method == 'preview_test_llms':
        preview_test_llms(model_name=args.model, mode=args.mode)
    elif args.method == 'evaluate_prediction':
        evaluate_prediction(model_name=args.model, mode=args.mode,
            check_unittest=args.check_unittest,
            check_indent=args.check_indent,
            check_cross_line=args.check_cross_line,
            check_empty=args.check_empty)
    elif args.method == 'print_scores':
        print_scores(model_name=args.model, mode=args.mode)
