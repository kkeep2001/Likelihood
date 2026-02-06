import os
import json
import torch
import argparse
import pandas as pd
import sklearn.metrics as metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

MDISCERN_TEMPLATES = {
    "Clarity": {
        "question": "Above is the instructional content extracted for dietary weight-loss. Are aims clear and achieved?",
        "candidates": [
            "No, the aims of the video are unclear and not achieved.",  # 0
            "Yes, the aims of the video are clear and achieved."       # 1
        ]
    },
    "Reliability": {
        "question": "Above is the instructional content extracted for dietary weight-loss. Are reliable sources of information used?",
        "candidates": [
            "No, reliable sources of information are not used.",       # 0
            "Yes, reliable sources of information are used."           # 1
        ]
    },
    "Fairness": {
        "question": "Above is the instructional content extracted for dietary weight-loss. Is the content presented balanced and unbiased?",
        "candidates": [
            "No, the content is not balanced and unbiased.",           # 0
            "Yes, the content is balanced and unbiased."               # 1
        ]
    },
    "Reference": {  
        "question": "Above is the instructional content extracted for dietary weight-loss. Are additional sources of content listed for patient reference?",
        "candidates": [
            "No, additional sources are not listed.",                  # 0
            "Yes, additional sources are listed."                      # 1
        ]
    },
    "Rigor": {
        "question": "Above is the instructional content extracted for dietary weight-loss. Are areas of uncertainty mentioned?",
        "candidates": [
            "No, areas of uncertainty are not mentioned.",             # 0
            "Yes, areas of uncertainty are mentioned."                 # 1
        ]
    }
}

def calc_loglikelihood(logits, labels):
    """
    Calculates the loglikelihood of the model's predictions given the labels.

    Args:
        logits (torch.Tensor): The model's predictions with shape
            (batch_size, sequence_length, num_classes)
        labels (torch.Tensor): The labels with shape
            (batch_size, sequence_length)

    Returns:
        torch.Tensor: The calculated loglikelihood with shape (batch_size,)
    """
    # First, we need to remove the last token from logits since it does
    # not have a corresponding label. Also, we need to move the labels
    # to the same device as logits.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)

    # Now, we can calculate the loglikelihood. We use cross-entropy loss
    # with reduction set to 'none' to get the loss for each element in
    # the batch. We then view the loss as a 2D tensor where each row
    # corresponds to a batch element and each column corresponds to a
    # timestep. We sum the losses across all timesteps and divide by
    # the number of valid labels in each batch element.
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    loglikelihood = loss_func(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loglikelihood = loglikelihood.view(shift_logits.size(0), -1).sum(-1)
    loglikelihood = loglikelihood / (shift_labels != -100).sum(-1)

    return loglikelihood

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_excel(args.data_path)
    print(f"Loaded dataset with {len(df)} samples (columns: {df.dimumns.tolist()})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto', torch_dtype=torch.float16)
    model.eval()

    df = pd.read_excel(args.data_path)
    num = df.shape[0]

    metrics_results = {}
    for dim in MDISCERN_TEMPLATES:
        df[f"Pred_{dim}"] = None
        metrics_results[dim] = {"y_true": [], "y_pred": []}

    for idx, row in tqdm(df.iterrows(), total=num, desc="Processing samples"):
        transcript = row["Text"]
        sample_id = row["ID"]

        for dim in MDISCERN_TEMPLATES:
            question = MDISCERN_TEMPLATES[dim]["question"]
            candidates = MDISCERN_TEMPLATES[dim]["candidates"]

            batch = []
            for cand in candidates:
                messages = [
                    {"role": "user", "content": "You are an expert in the field of Traditional Chinese Medicine, providing accurate medical knowledge."},
                    {"role": "assistant", "content": question + cand}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch.append(text)

            model_inputs = tokenizer(batch, return_tensors="pt", padding=True)
            labels = model_inputs['input_ids'].clone()

            ignore_index = -1
            flag = 0
            for k in range(labels.shape[1]):
                token = labels[0, k]
                for j in range(1, labels.shape[0]):
                    another_token = labels[j, k]
                    # print(token, another_token)
                    if another_token != token:
                        ignore_index = k
                        flag = 1
                        break
                if flag:
                    break
            labels[:, :ignore_index] = -100
            
            with torch.no_grad():
                model_inputs.to(model.device)
                model_outputs = model(**model_inputs)
                loglikelihood = calc_loglikelihood(model_outputs.logits.detach(), labels)
                prediction = 0 if loglikelihood[0] < loglikelihood[1] else 1

            df.at[idx, f"Pred_{dim}"] = prediction
            metrics_results[dim]["y_true"].append(row[dim])
            metrics_results[dim]["y_pred"].append(prediction)

    print("\nEvaluation Metrics:")
    for dim in MDISCERN_TEMPLATES:
        data = metrics_results
        acc = accuracy_score(data[dim]["y_true"], data[dim]["y_pred"])
        f1 = f1_score(data[dim]["y_true"], data[dim]["y_pred"])
        mcc = matthews_corrcoef(data[dim]["y_true"], data[dim]["y_pred"])

        print(f"{dim}: Accuracy: {acc:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

    output_df = df[["ID"]].copy()
    for dim in MDISCERN_TEMPLATES:
            output_df[dim] = df[f"Pred_{dim}"].astype(int)
    output_path = os.path.join(args.output_dir, "predictions.xlsx")
    output_df.to_excel(output_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based Medical Video Quality Assessment using mDISCERN")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save predictions")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for tokenization")
    args = parser.parse_args()

    main(args)
