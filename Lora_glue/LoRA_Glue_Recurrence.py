import sys
from datasets import load_dataset, load_metric, concatenate_datasets
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, \
    DataCollatorWithPadding
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import evaluate

# dataset_name : mrpc、cola、qnli、qqp、rte、stsb、mnli、sst2、
dataset_name = 'mnli'
# model_name : flan-t5-xxl,flan-t5-base,flan-t5-large,roberta-large,bert-large-uncased,bert-large-cased,gpt-2,gpt-3
model_name = 'roberta-large'
# glue
dataset_series_name = 'glue'

output_dir = f"./output/yg1997/{model_name}/{dataset_name}".replace('/', os.sep)
data_dir = f'./data/yg1997/{dataset_series_name}/{dataset_name}'.replace('/', os.sep)
model_dir = f'./model/{model_name}'.replace('/', os.sep)

learning_rate = 1e-4
epochs = 20

# 记录：
# 已完成：mrpc,cola,rte,sst2,stsb,qnli,mnli,qqp
# 待完成：(二分类)mrpc,cola,rte,sst2,qqp,qnli
#       (多分类)mnli(3)
#       (回归任务)stsb
num_lables = 1 if dataset_name in ['stsb'] else (3 if dataset_name in ['mnli'] else 2)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# print("使用的 GPU 数量:", n_gpu)

def get_model_tokenizer():
    print(f'开始导入{model_name}模型')
    model = None
    if model_name == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=resume_download_config)
        if dataset_name in ['mrpc', 'cola', 'rte', 'sst2', 'qqp', 'qnli', 'stsb', 'mnli']:
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       cache_dir=model_dir,
                                                                       # load_in_8bit=True,
                                                                       device_map={
                                                                           "": int(os.environ.get("LOCAL_RANK") or 0)},
                                                                       # device_map="auto",
                                                                       resume_download=resume_download_config,
                                                                       num_labels=num_lables)
    else:
        print("错误的模型名称，获取模型失败，退出！")
        sys.exit(0)

    # 进行整数量化训练，以获得整数量化后的模型，从而提高模型在边缘设备上的推理效率
    # 即进行整数量化训练，以获得整数量化后的模型，从而提高模型在边缘设备上的推理效率
    # INT8训练通过将权重和激活值压缩为8位整数，而不是浮点数(模型通常使用浮点精度（如32位或16位）来表示权重和激活)
    # 来提供更高的计算效率和内存利用率。
    # model = prepare_model_for_int8_training(model)
    # 配置lora参数
    print('开始配置lora参数')
    lora_config = get_lora_config(dataset_name)
    # 将lora和base模型结合
    print('开始将lora和base结合')
    model = get_peft_model(model, lora_config)
    print('模型处理完成')
    return model, tokenizer


def get_lora_config(dataset_name):
    """
    根据任务的不同，得到不同的lora_config
    SEQ_CLS="SEQ_CLS" （序列分类，Sequence Classification）：
                      这种任务类型要求将输入序列分类为预定义的几个类别之一。这种任务通常用于文本分类、
                      情感分析等场景。模型的输出是一个类别标签。
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM" （序列到序列的语言建模，Sequence-to-Sequence Language Modeling）：
                                   这种任务类型要求将输入序列翻译或转换为另一个序列。这种任务通常用于机器翻译、
                                   文本生成等场景。模型的输出是生成的序列。
    CAUSAL_LM = "CAUSAL_LM" （因果语言建模，Causal Language Modeling）：
                              这种任务类型要求根据输入的部分序列预测下一个单词或词语。
                              这种任务通常用于自动补全、语言生成等场景。模型的输出是下一个预测的单词或词语。
    TOKEN_CLS = "TOKEN_CLS" （令牌分类，Token Classification）：
                             这种任务类型要求将这一分序列中的消息令牌标记为预定位置的几中类别以上。
                             这种任务类型要求将输入序列中的每个令牌标记为预定义的几个类别之一。
                             这种任务通常用于命名实体识别、词性标注等场景。
                             模型的输出是对输入序列中每个令牌进行的分类标签。
    """
    if dataset_name in ['mrpc', 'cola', 'rte', 'sst2', 'qqp', 'qnli', 'stsb', 'mnli']:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            # 即论文图中右边两个模块输出的dropout
            lora_dropout=0.05,
            bias="none",
            # 说明任务类型，从而影响模型的架构，loss，输出形式
            task_type=TaskType.SEQ_CLS
        )
    else:
        print("错误的数据集名称，lora_config生成失败，退出！")
        sys.exit(0)

    return lora_config


def get_dataset():
    if dataset_name not in ['mnli', 'sst2', 'mrpc', 'cola', 'qnli', 'qqp', 'rte', 'stsb']:
        print("错误数据集名称,下载数据集失败，退出！")
        sys.exit(0)
    print(f'开始下载{dataset_name}数据集。。。')
    dataset = load_dataset('glue', dataset_name, cache_dir=data_dir,
                           download_config=resume_download_config)
    print(f"训练集数据大小: {len(dataset['train'])}")
    # print(f"测试集数据大小: {len(dataset['validation' if dataset_name != 'mnli' else 'validation_matched'])}")
    print('数据集下载完成！')
    return dataset


def preprocess_dataset(dataset):
    # tokenization期间有足够的 RAM 来存储整个数据集时这种方法才有效
    # （而 🤗 Datasets 库中的数据集是存储在磁盘上的 Apache Arrow 文件，因此您只需将请求加载的样本保存在内存中）。
    # 所以最好是用dataset.map
    if dataset_name in ['mrpc', 'rte', 'stsb']:
        # https://blog.csdn.net/qq_56591814/article/details/120147114
        # 上述博客写的比较详细
        # 保留lable列，后续测试要用到
        dataset = dataset.map(mrpc_rte_sstb_preprocess_function,
                              batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    elif dataset_name in ['cola', 'sst2']:
        dataset = dataset.map(cola_sst2_preprocess_function,
                              batched=True, remove_columns=["sentence", "idx"])
    elif dataset_name in ['qqp']:
        dataset = dataset.map(qqp_preprocess_function,
                              batched=True, remove_columns=["question1", "question2", "idx"])
    elif dataset_name in ['qnli']:
        dataset = dataset.map(qnli_preprocess_function,
                              batched=True, remove_columns=["question", "sentence", "idx"])
    elif dataset_name in ['mnli']:
        dataset = dataset.map(mnli_preprocess_function,
                              batched=True, remove_columns=["premise", "hypothesis", "idx"])
    else:
        print("错误数据集名称,数据预处理失败，退出！")
        sys.exit(0)

    print(f"\n编码后数据集的特征: {list(dataset['train'].features)}")
    # print('开始保存编码后数据集。。。')
    # dataset["train"].save_to_disk(f"{output_dir}/train".replace('/', os.sep))
    # dataset["validation"].save_to_disk(f"{output_dir}/eval".replace('/', os.sep))
    # print('编码后数据集保存完毕')
    return dataset


def mrpc_rte_sstb_preprocess_function(example):
    # 会自动添加起始结束和分隔符
    # ['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
    # [      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
    # tokenization函数中省略了padding 参数，这是因为padding到该批次中的最大长度时的效率，
    # 会高于所有序列都padding到整个数据集的最大序列长度。 当输入序列长度很不一致时，这可以节省大量时间和处理能力！
    input = tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    return input


def qqp_preprocess_function(example):
    input = tokenizer(example["question1"], example["question2"], truncation=True)
    return input


def qnli_preprocess_function(example):
    input = tokenizer(example["question"], example["sentence"], truncation=True)
    return input


def mnli_preprocess_function(example):
    input = tokenizer(example["premise"], example["hypothesis"], truncation=True)
    return input


def cola_sst2_preprocess_function(example):
    # ['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]']
    # [      0,      0,    0,     0,       0,          0,   0]
    # tokenization函数中省略了padding 参数，这是同样的道痛。
    input = tokenizer(example["sentence"], truncation=True)
    return input


def get_data_collator():
    if dataset_name in ['mrpc', 'cola', 'rte', 'sst2', 'qqp', 'qnli', 'stsb', 'mnli']:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    else:
        print("错误数据集名称,数据收集器生成失败，退出！")
        sys.exit(0)
    return data_collator


def get_training_args():
    eval_save_steps = 10000 if dataset_name in ['mnli', 'qqp', 'qnli'] else 1000
    if dataset_name in ['mrpc', 'cola', 'rte', 'sst2', 'qqp', 'qnli', 'stsb', 'mnli']:
        training_args = TrainingArguments(
            output_dir=output_dir,
            # 设置为True时，会自动寻找适合的批次大小。
            auto_find_batch_size=True,
            # 在每个epoch的步数后进行验证评估
            evaluation_strategy="steps",
            # 设置评估的步数间隔
            eval_steps=eval_save_steps,
            # 设置保存模型的步数间隔
            save_steps=eval_save_steps,
            # 设置保存的模型数量上限
            save_total_limit=10,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            # 梯度累积，可以帮助处理显存不足的情况
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            # 学习率预热的步数
            warmup_steps=500,
            # 训练日志存储目录
            logging_dir=f"{output_dir}/logs".replace('/', os.sep),
            # 训练日志输出的步数间隔
            logging_steps=1000,
            # 是否禁用tqdm进度条
            disable_tqdm=False,
            # 训练结束后自动加载最佳模型
            load_best_model_at_end=True,
            seed=2023,
            report_to=["tensorboard"],
        )
    else:
        print("错误数据集名称,训练参数设置失败，退出！")
        sys.exit(0)
    return training_args


def get_trainer(training_args, tokenized_dataset, data_collator):
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation" if dataset_name != 'mnli' else 'validation_matched'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer


def compute_metrics(eval_preds):
    """
    compute_metrics 函数必须传入一个 EvalPrediction 对象作为参数。 EvalPrediction是一个具有预测字段和 label_ids 字段的元组。
    compute_metrics返回的结果是字典，键值对类型分别是strings和floats（strings是metrics的名称，floats是具体的值）。
    直接调用metric的compute方法，传入labels和predictions即可得到metric的值。也只有这样做才能在训练时得到acc、F1（F1 = 2 * (精确度 * 召回率) / (精确度 + 召回率)）
    等结果（具体指标根据不同任务来定）
    """
    if dataset_name in ['mrpc', 'cola', 'rte', 'sst2', 'qqp', 'qnli', 'stsb', 'mnli']:
        # 根据数据集不同，下载不同的metric
        metric = evaluate.load("glue", f'{dataset_name}', download_config=resume_download_config)
        # metric = load_metric("glue", f'{dataset_name}', download_config=resume_download_config)
        # 拿出一个测试对象的eval_preds，变量包含了模型在测试集上的预测结果和对应的标签。
        # logits表示模型对每个样本的预测结果，labels是样本对应的真实标签。
        # mrpc中logits是（408,2）的数组，类似于下面的数组，每一行的两列，分别表示预测为第一类的概率和第二类的概率。即第一行预测两个句子相关的概率为0.8
        # logits = [[-1.2, 0.8],
        #           [0.5, -0.3],
        #           [1.1, 2.4]]
        logits, labels = eval_preds
        # 使用np.argmax函数获取每个样本的最大预测值的索引，即预测类别。这里假设预测结果是一个概率分布，通过取概率最高的类别作为最终预测结果。
        # 得到的结构为[1, 0, 1]，即每一行取最大值所在的位置，第一行0.8更大，所以为1，以此类推
        # 对于stsb这种回归任务，算相似度需要将logits.squeeze(-1)，其中logits(n,1),lable(n,)。
        # 然后直接计算相似度，返回{'pearson': float_num, 'spearmanr': float_num}
        predictions = logits.squeeze(-1) if dataset_name in ['stsb'] else np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    else:
        print("错误数据集名称,获取metric失败，退出！")
        sys.exit(0)


def train(tokenized_dataset):
    data_collator = get_data_collator()
    training_args = get_training_args()
    trainer = get_trainer(training_args, tokenized_dataset, data_collator)
    print("开始训练")
    trainer.train()

    print("训练完毕，保存数据")
    # save_path = f"{output_dir}/results".replace('/', os.sep)
    # trainer.model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    print("数据保存完毕")


if __name__ == '__main__':
    try:
        # 科学上网问题，重复进行下载，避免下载失败退出
        resume_download_config = datasets.DownloadConfig(resume_download=True, max_retries=300)
        # 获取model,tokenizer
        model, tokenizer = get_model_tokenizer()
        # 获取数据集
        dataset = get_dataset()
        # 预处理数据
        tokenized_dataset = preprocess_dataset(dataset)
        # 开始训练
        train(tokenized_dataset)
        print('程序完成，退出！')
        sys.exit(0)
    except SystemExit:
        sys.exit(0)
