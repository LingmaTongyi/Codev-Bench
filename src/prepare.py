# -*- coding: utf-8 -*-
# @Author: persistforever
# @Time: 2024/09/19
"""
使用前的准备工作
"""
import os
import argparse
import tqdm
import json
import pandas
import re
import pathlib
import shutil
import random


MAIN_DIR = '/mnt/coai_nas/qianhu/github/Codev-Bench/'


def get_tree_structure(blocks):
    """
    获取blocks的树结构
    """
    relation_dict = {
        'root': {'parent': None, 'children': [0]},
        0: {'parent': 'root', 'children': []},
    }
    stack = ['root', 0]
    for index in range(1, len(blocks)):
        current_block = blocks[index]
        relation_dict[index] = {'parent': None, 'children': []}
        top = stack[-1]
        # 如果当前block的startOffset大于栈顶的endOffset
        #   则出栈直到找到一个block的endOffset大于当前block的startOffset
        #   或者找到root节点
        if current_block['endOffset'] > blocks[top]['endOffset'] :
            while top != 'root':
                if blocks[top]['endOffset'] >= current_block['endOffset'] :
                    break
                del stack[-1]
                top = stack[-1]
            relation_dict[top]['children'].append(index)
            relation_dict[index]['parent'] = top
            stack.append(index)
        # 否则，将当前节点添加到栈顶节点的children中
        else:
            relation_dict[stack[-1]]['children'].append(index)
            relation_dict[index]['parent'] = stack[-1]
            stack.append(index)

    # 寻找每个block的上一个block
    for bid in relation_dict:
        current_block = relation_dict[bid]
        if bid == 'root':
            current_block['prev_sibling'] = None
            continue
        parent_block = relation_dict[current_block['parent']]
        child_position = parent_block['children'].index(bid)
        # 如果是第一个child，则没有prev_sibling，否则找到前一个兄弟作为prev_sibling
        if child_position == 0:
            current_block['prev_sibling'] = None
        else:
            current_block['prev_sibling'] = parent_block['children'][child_position - 1]
        # 如果是最后一个child，则没有next_sibling，否则找到后一个兄弟作为next_sibling
        if child_position == len(parent_block['children']) - 1:
            current_block['next_sibling'] = None
        else:
            current_block['next_sibling'] = parent_block['children'][child_position + 1]
    
    # 设置祖先节点、子孙节点
    for bid in relation_dict:
        relation_dict[bid]['ancestor'] = {}
        relation_dict[bid]['descendant'] = {}
    for bid in relation_dict:
        current_bid = relation_dict[bid]['parent']
        while current_bid is not None:
            relation_dict[current_bid]['descendant'][bid] = None
            relation_dict[bid]['ancestor'][current_bid] = None
            current_bid = relation_dict[current_bid]['parent']

    return relation_dict


def run_unit_test_command(item, log_path, mode='normal'):
    """
    运行单测
    """
    patterns = [r'===== (\d+) passed in ', r'===== (\d+) passed, \d+ warning in ', r'===== (\d+) passed, \d+ warnings in ']
    if mode == 'normal':
        save_command = f"{item['run_command']} --timeout=120 --timeout_method=thread > '{log_path}'"
    elif mode == 'trace':
        save_command = f"{item['run_command']} --timeout=120 --timeout_method=thread -p pytest_trace_calls --trace-calls > '{log_path}'"
    # print(save_command)
    os.system(save_command)
    lines = []
    with open(log_path, 'r') as fo:
        for line in fo:
            lines.append(line.strip())
    last_line = ''
    for line in lines[::-1]:
        if len(line.strip()) != 0:
            last_line = line
            break
    for pattern in patterns:
        results = re.findall(pattern, last_line)
        if len(results) != 0 and int(results[0]) >= 1:
            return True, ''
    return False, '\n'.join(lines)


def filter_method_dict(cosine_path, source_code_dir):
    """
    过滤出来所有method的block
    """
    # 获取文件名->blocks的字典
    method_line_dict, method_dict, block_dict = {}, {}, {}
    df = pandas.read_csv(cosine_path)
    # 获取block的元信息
    for info in tqdm.tqdm(df.iloc, desc='read block info'):
        file_absolute_path = os.path.join(source_code_dir, info['file_path'])
        blocks = json.loads(info['block'])['blocks']
        bid2bkey = {}
        for bid, block in enumerate(blocks):
            block_key = f"{file_absolute_path}#L{block['startLine']+1}-L{block['endLine']+1}"
            block['block_key'] = block_key
            block['bid'] = str(bid)
            block['block_code'] = '\n'.join(info['file_content'].split('\n')[
                block['startLine']: block['endLine']+1])
            if 'header' in block and block['header'] is not None:
                block['header_code'] = '\n'.join(info['file_content'].split('\n')[
                    block['header']['startLine']: block['header']['endLine']+1])
                indent = ''
                for line in block['block_code'][len(block['header_code']):].split('\n'):
                    if len(line.strip()) != 0:
                        for char in line:
                            if len(char.strip()) == 0:
                                indent += char
                            else:
                                break
                        break
                block['masked_block_code'] = block['header_code'] + '\n' + indent + 'pass'
            block_dict[block['block_key']] = block
            bid2bkey[bid] = block['block_key']
        # type预处理
        for block in blocks:
            if block['type'] == 'EXPRESSION':
                if (block['block_code'].lstrip().startswith('"""') and \
                        block['block_code'].rstrip().endswith('"""')) or \
                        (block['block_code'].lstrip().startswith("'''") and \
                        block['block_code'].rstrip().endswith("'''")):
                    block['type'] = 'BLOCK_COMMENT'
                else:
                    block['type'] = 'STATEMENT'
        # 恢复block的ast树中的路径
        relation_dict = get_tree_structure(blocks)
        for bid, block in enumerate(blocks):
            if block['type'] == 'METHOD':
                node_path = [block['name'].split('.')[-1].split('(')[0]]
                current_node = relation_dict[bid]['parent']
                while current_node != 'root':
                    parent_block = blocks[int(relation_dict[bid]['parent'])]
                    name = parent_block['name'].split('.')[-1].split('(')[0]
                    node_path.append(name)
                    current_node = relation_dict[current_node]['parent']
                block['node_path'] = '::'.join(node_path[::-1])
                # 如果header结尾不是“:”则过滤掉
                if not block['header_code'].rstrip().endswith(':'):
                    continue
                # 如果header不是code的开头，则过滤掉
                if not block['block_code'].startswith(block['header_code']):
                    continue
                block['descendant'] = {}
                for desc_bid in relation_dict[bid]['descendant']:
                    block['descendant'][bid2bkey[desc_bid]] = None
                method_dict[block['block_key']] = block
                for lineno in range(block['startLine']+1, block['endLine']+2):
                    method_line_dict[f"{file_absolute_path}#L{lineno}"] = block['block_key']
    
    return method_line_dict, method_dict, block_dict


def retest_block_unit_test(mode='prefix_suffix_full_complete_current_block_no_evidence'):
    """
    获取测试用例列表，重新测试
    """
    main_dir = MAIN_DIR
    source_code_dir = os.path.join(main_dir, 'Source_Code')
    copyed_source_code_dir = os.path.join(main_dir, 'Source_Code_Copy')
    meta_dir = os.path.join(main_dir, 'metadatas/')
    cosine_path = os.path.join(main_dir, 'metadatas/test_file_cosine.output.csv')
    test_path = os.path.join(main_dir, 'prompts', f'{mode}.jsonl')
    
    # 获取所有挖空目标位置
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

    # 寻找所有单测
    unit_test_id_dict = {}
    for block_key in tqdm.tqdm(list(block_dict.keys()), desc='find ut'):
        block_info = block_dict[block_key]
        block_key = block_key.replace('/mnt/coai_nas/qianhu/github/completion_benchmark/', MAIN_DIR)
        file_path = block_key.split('#')[0]
        repo_name = file_path[len(source_code_dir)+1:].split('/')[0]
        log_dir = os.path.join(meta_dir, repo_name, 'retest_block_unit_tests')
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        for unit_test_id in block_info['unit_test_ids']:
            unit_test_id_dict[unit_test_id] = [file_path, repo_name, log_dir]
    
    # 运行所有单测
    n_all, n_passed = 0, 0
    for unit_test_id in tqdm.tqdm(list(unit_test_id_dict.keys()), desc='run ut'):
        [file_path, repo_name, log_dir] = unit_test_id_dict[unit_test_id]
        test_item = test_dict[unit_test_id]
        # 要被修改的文件复制
        source_path = file_path.replace('Source_Code', 'Source_Code_Copy')
        target_path = file_path
        shutil.copy(source_path, target_path)
        # 运行原始的单测
        log_path = os.path.join(
            log_dir, repo_name,
            f"{test_item['relative_path'].replace('/', '!')}::{test_item['func_path']}.original.txt")
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        is_origin_pass, _ = run_unit_test_command(test_item, log_path)
        # 之前通过之后不通过，则说明是候选函数
        n_all += 1
        if is_origin_pass:
            n_passed += 1
        print(f"run {n_all} unit tests, {n_passed} is passed, " \
            f"pass rate is {100.0*n_passed/n_all:.2f}%")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, 
        help='The method needed to run.',
        default='retest_block_unit_test')
    parser.add_argument("--mode", type=str,
        help='The sub-scene will be evaluated.',
        default='prefix_suffix_full_complete_current_block_no_evidence')
    args = parser.parse_args()
    if args.method == 'retest_block_unit_test':
        retest_block_unit_test()
