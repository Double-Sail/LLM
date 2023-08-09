import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def eval():
    # 为预先训练的检查点加载peft配置等。
    save_path = "/output/results"
    config = PeftConfig.from_pretrained(save_path)

    # load base LLM model and tokenizer
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, save_path, device_map={"": 0})
    model.eval()

    print("模型载入完毕")

    # Metric
    # ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是一种常用的自动文本摘要评估指标，用于衡量生成的摘要与参考摘要之间的相似程度。
    # Rouge1	基于单元词组(unigram)的评估
    # Rouge2	基于二元词组(bigram)的评估
    # RougeL	基于最长公共子序列(LCS, Longest common subsequence)的评估
    # RougeS, RougeW, RougeSW	不常用，可见自动文摘评测方法：Rouge-1、Rouge-2、Rouge-L、Rouge-S
    metric = evaluate.load("rouge")

    def evaluate_peft_model(sample, max_target_length=50):
        # generate summary
        # 使用加载的模型对输入进行生成（generation）。generate函数采用input_ids作为输入，并生成最多10个新的令牌（tokens）。
        # 设置do_sample=True表示采样生成，设置top_p=0.9表示采样时保留总概率最高的90%的词汇
        outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9,
                                 max_new_tokens=max_target_length)
        prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        # decode eval sample
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
        labels = tokenizer.decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        return prediction, labels

    # load test dataset from distk
    test_dataset = load_from_disk("/output/data/eval").with_format("torch")

    # run predictions
    # this can take ~45 minutes
    predictions, references = [], []
    # tqdm函数，用于在循环中显示进度条，以便可视化地跟踪评估的进度
    for sample in tqdm(test_dataset):
        p, l = evaluate_peft_model(sample)
        predictions.append(p)
        references.append(l)

    # compute metric
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    # print results
    print(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
    print(f"rouge2: {rogue['rouge2'] * 100:2f}%")
    print(f"rougeL: {rogue['rougeL'] * 100:2f}%")
    print(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")

    # Rogue1: 50.386161%
    # rouge2: 24.842412%
    # rougeL: 41.370130%
    # rougeLsum: 41.394230%


if __name__ == '__main__':
    eval()
