import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def calc_loglikelihood(logits, labels):

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)

    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loglikelihood = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loglikelihood = loglikelihood.view(shift_logits.size(0), -1).sum(-1)
    loglikelihood = loglikelihood / (shift_labels != -100).sum(-1)
    return loglikelihood

if __name__ == '__main__':
    
    model_dir = 'path/model_directory'
    src_dir = 'path/source_directory'
    name = 'path/input.json'
    dst_dir = 'path/destination_directory'
    excel_output_path = os.path.join(dst_dir, 'output.xlsx')

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    df = pd.read_json(os.path.join(src_dir, name))
    num = df.shape[0]

    res = [''] * num
    logits_res = [[] for _ in range(num)]
    dic = {0: 'A', 1: 'B'}

    for i in tqdm(range(num), desc="Processing questions"):
        question = df.iloc[i, 0]
        options = df.iloc[i, 2:4].tolist()
        batch = []

        for op in options:
            if not op:
                continue
            messages = [
                {"role": "system", "content": '你是一个医疗文本质量评分助手，帮我从五个方面对文本质量进行评估。'},
                {"role": "user", "content": question},
                {"role": "assistant", "content": op}
            ]
            text_best = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch.append(text_best)

        model_inputs = tokenizer(batch, return_tensors="pt", padding=True)
        labels = model_inputs['input_ids'].clone()

        ignore_index = -1
        flag = 0
        for k in range(labels.shape[1]):
            token = labels[0, k]
            for j in range(1, labels.shape[0]):
                if labels[j, k] != token:
                    ignore_index = k
                    flag = 1
                    break
            if flag:
                break
        labels[:, :ignore_index] = -100

        model_inputs.to(model.device)
        with torch.no_grad():
            model_outputs = model(**model_inputs)

        loss = calc_loglikelihood(model_outputs.logits.detach(), labels)
        res[i] = dic[loss.argmin().item()]
        logits_res[i] = loss.cpu().tolist()

    df['Prediction'] = res
    df['LogLikelihoods'] = logits_res
    df.to_json(os.path.join(dst_dir, name), orient="records", indent=2, force_ascii=False)

    group_size = 5
    num_groups = (len(df) + group_size - 1) // group_size
    group_results = []

    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, len(df))

        y_true_group = df['Ans'][start_idx:end_idx]
        y_pred_group = df['Prediction'][start_idx:end_idx]
        logits_group = logits_res[start_idx:end_idx]

        group_accuracy = [1 if t == p else 0 for t, p in zip(y_true_group, y_pred_group)]
        accuracy = sum(group_accuracy) / len(group_accuracy) if len(group_accuracy) > 0 else 0

        group_row = [group_idx + 1] + group_accuracy

        for ll in logits_group:
            group_row += ll

        for _ in range(group_size - len(logits_group)):
            group_row += [None, None]
        group_row.append(accuracy)
        group_results.append(group_row)


    columns = ['column'] + [f'Question{i+1}' for i in range(group_size)]
    for q in range(group_size):
        columns += [f'Question{q+1}_OptionA_LogL', f'Question{q+1}_OptionB_LogL']
    columns.append('ACC')

    group_df = pd.DataFrame(group_results, columns=columns)
    group_df.to_excel(excel_output_path, index=False)


    correct_count = sum([a == b for a, b in zip(df['Ans'], df['Prediction'])])
    total_count = len(df)
    accuracy = correct_count / total_count
    print(f"Overall Correct Count: {correct_count}/{total_count}, Overall Accuracy: {accuracy:.2f}")
    
