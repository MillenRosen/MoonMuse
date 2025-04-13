import time

import scipy
import torch
import numpy as np

from utils import tensor_to_numpy
from convert_key import MINOR_KEY, MAJOR_KEY


########################################
# 采样工具函数
########################################

def temperature(logits, temperature):
    """
    使用温度参数对logits进行缩放，并计算概率分布。
    
    参数:
    - logits: 模型的原始输出（未归一化的对数概率）。
    - temperature: 温度参数，控制概率分布的平滑程度。温度越高，分布越平滑；温度越低，分布越尖锐。
    
    返回:
    - probs: 经过温度缩放后的概率分布。
    """
    try:
        # 使用温度参数缩放logits，并计算softmax得到概率分布
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        # 检查是否有NaN值
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        # 如果出现溢出，使用128位浮点数进行计算
        print('overflow detected, use 128-bit')
        logits = logits.astype(np.float128)
        probs = scipy.special.softmax(logits / temperature)
        probs = probs.astype(float)
        # 再次检查是否有NaN值
        assert np.count_nonzero(np.isnan(probs)) == 0
    return probs


def nucleus(probs, p):
    """
    使用nucleus sampling（也称为top-p sampling）从概率分布中采样。
    
    参数:
    - probs: 概率分布。
    - p: 累积概率阈值，只从累积概率超过p的最小集合中采样。
    
    返回:
    - word: 采样得到的索引。
    """
    # 归一化概率分布
    probs /= sum(probs)
    # 对概率进行降序排序
    sorted_probs = np.sort(probs)[::-1]
    # 获取排序后的索引
    sorted_index = np.argsort(probs)[::-1]
    # 计算累积概率
    cusum_sorted_probs = np.cumsum(sorted_probs)
    # 找到累积概率超过p的索引
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        # 获取第一个超过p的索引
        last_index = np.where(after_threshold)[0][1]
        # 候选索引为累积概率超过p的最小集合
        candi_index = sorted_index[:last_index]
    else:
        # 如果没有超过p的索引，则默认选择前3个
        candi_index = sorted_index[:3]  # 默认赋值
    # 获取候选索引对应的概率
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    # 归一化候选概率
    candi_probs /= sum(candi_probs)
    # 从候选索引中根据概率分布采样一个索引
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def get_position_idx(event):
    """
    从事件名称中提取位置索引。
    
    参数:
    - event: 事件名称，格式为"event_position"。
    
    返回:
    - position_idx: 事件名称中的位置索引。
    """
    return int(event.split('_')[-1])

########################################
# main inference driver
########################################
def generate_plain_xl(model, event2idx, idx2event, max_bars=160,
                      max_events=2048, primer=None, temp=1.2, top_p=0.9,
                      prompt_bars=None, representation='functional', key_determine=None):
    if primer is None: # 无先前生成的音符
        generated = [event2idx['Bar_None']]
        # print('[test 1] generated:', generated)
        target_bars, generated_bars = max_bars, 0
    else: # 有先前生成的音符
        generated = [event2idx[e] for e in primer]
        # print('[test 2] generated:', generated)
        target_bars, generated_bars = max_bars, prompt_bars if prompt_bars is not None else 0

    device = next(model.parameters()).device
    steps = 0 # 计步数
    time_st = time.time() # 计时
    cur_pos = 0 # 当前音符位置
    failed_cnt = 0 # 连续失败次数
    mems = tuple() # 记忆单元
    while generated_bars < target_bars: # 循环生成音符
        if steps == 0: # 首次生成
            dec_input = torch.LongTensor([generated]).to(device) 
            dec_input = dec_input.permute(1, 0) if len(generated) > 1 else dec_input 
            # print ('[shape]', dec_input.size(), dec_seg_emb.size())
        else:
            dec_input = torch.LongTensor([[generated[-1]]]).to(device)
        # print (dec_input.size(), dec_seg_emb.size())
        # print('[test 4] dec_input:', dec_input)

        # sampling
        logits, mems = model.generate(dec_input, mems) # 生成下一个token的logits
        # print('[test 5] logits:', logits)
        # print('[test 5] mems[0].size():', mems[0].size())
        logits = tensor_to_numpy(logits)

        if representation in ['functional', 'key'] and len(generated) == 1:
            probs = temperature(logits, temperature=1.1) # probs为当前token的概率分布
            word = nucleus(probs, p=0.97) # 取概率分布的top_p个token
            # print('[test] word:', word, idx2event[word])
            if key_determine == 'rule':
                emotion_label = idx2event[generated[0]].split('_')[1] 
                key_event = idx2event[word] 
                if key_event.split('_')[0] != 'Key':
                    raise ValueError('[info] key generation failed')
                key_label = key_event.split('_')[1] 
                if not match_emotion_key(emotion_label, key_label):
                    continue
            word_event = idx2event[word] 
            # print('[test] word_event:', word_event)
        else:
            probs = temperature(logits, temperature=temp)
            word = nucleus(probs, p=top_p)
            # print('[test] word(else):', word)
            word_event = idx2event[word]
            # print('[test] word_event(else):', word_event)
            # print(temp, top_p)

        if 'Key' in word_event:
            print('[info] generated {}, #events = {}'.format(word_event, len(generated)))

        if 'Beat' in word_event:
            event_pos = get_position_idx(word_event)
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt, end='\r')
                if failed_cnt >= 256:
                    print('[FATAL] model stuck, exiting ...')
                    return None, time.time() - time_st
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        if 'Bar' in word_event:
            generated_bars += 1
            cur_pos = 0
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
        if word_event == 'PAD_None':
            continue

        generated.append(word)
        # print('[test 3] generated:', generated)
        # print ([idx2event[e] for e in generated])
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break
        if word_event == 'EOS_None':
            print('[info] gotten eos')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))

    return generated[:-1], time.time() - time_st


def match_emotion_key(emotion, key):
    if emotion in ['Q1', 'Q4', 'Positive'] and key in MAJOR_KEY:
        return True
    if emotion in ['Q2', 'Q3', 'Negative'] and key in MINOR_KEY:
        return True
    return False
