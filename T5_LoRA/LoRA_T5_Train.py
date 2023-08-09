import sys
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import concatenate_datasets
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# model_id = "google/flan-t5-xxl"
# model_id = "philschmid/flan-t5-xxl-sharded-fp16"
# model_id = "google/flan-t5-large"
model_id = "google/flan-t5-base"
dataset_name = "samsum"
label_pad_token_id = -100
output_dir = "/output"
data_dir = '/data/yg1997'
model_dir = '/model/yg1997/'


def get_model():
    """
    获取模型
    """
    # load_in_8bit：一个布尔值，用于指定加载模型时是否使用 8 位整数精度，即使用 8 位整数精度进行加载，可以降低模型的内存消耗。
    # device_map：一个字符串，指定模型在不同设备上的分布。可选值为 “auto”、“cpu” 或具体的设备名称（如 “cuda:0”）。
    # 默认为 “auto”，表示根据可用的硬件自动选择设备。
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=f'{model_dir}/{model_id}', load_in_8bit=True,
                                                  device_map="auto",
                                                  resume_download=resume_download_config)
    # 进行整数量化训练，以获得整数量化后的模型，从而提高模型在边缘设备上的推理效率
    # 即进行整数量化训练，以获得整数量化后的模型，从而提高模型在边缘设备上的推理效率
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        # 即论文图中右边两个模块输出的dropout
        lora_dropout=0.05,
        bias="none",
        # 说明任务类型，从而影响模型的架构，loss，输出形式
        # TaskType.SEQ_2_SEQ_LM（序列到序列语言建模任务）
        # TaskType.SENT_CLASSIFICATION（文本分类任务）
        # TaskType.QUESTION_ANSWERING（问答任务）
        # 。。。
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # 将lora和base模型结合，
    model = get_peft_model(model, lora_config)
    # trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817
    # 用于打印模型中可训练的参数的详细信息
    # 列出所有可训练的参数，并显示它们的名称、形状和数量。
    model.print_trainable_parameters()
    return model


def get_dataset():
    """
    获取数据集，这里用的sansum数据集，用于文本摘要

    samsum的数据结构
    {"id": "13818513",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
    "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)"
    }
    """
    # Hugging Face Datasets 库中的一个类，用于配置数据集的下载行为。
    # resume_download 参数，设置为 True。这意味着如果下载被中断，它将尝试继续从上次中断的地方恢复下载。
    # max_retries 参数设置为 100。这表示如果下载失败，它将进行最多 100 次重试，以便尝试重新下载数据集
    # 因为hugging face由于科学上网问题，会下载失败，一般需要手动重复下载数据几次，这个config操作，可以自动重复下载
    dataset = load_dataset(dataset_name, cache_dir=f'{data_dir}/{dataset_name}',
                           download_config=resume_download_config)
    # Train dataset size: 14732
    # Test dataset size: 819
    print(f"训练集数据大小: {len(dataset['train'])}")
    print(f"测试集数据大小: {len(dataset['test'])}")
    return dataset


def preprocess_dataset(dataset):
    # 经过预处理之后的结果---(max_source_length, max_target_length, dataset)
    tokenized_dataset = dataset.map(preprocess_function,
                                    batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"\nKeys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # 保存数据
    tokenized_dataset["train"].save_to_disk(f"{output_dir}/data/train")
    tokenized_dataset["test"].save_to_disk(f"{output_dir}/data/eval")
    return tokenized_dataset


def preprocess_dataset_truncate_padding(dataset):
    """
    将dialogue和summary进行长度限制，dialogue取长度的85%，summary取长度的90%
    tokenized_inputs和tokenized_targets先将训练集和测试集concatenate起来，然后分别对dialogue和summary进行tokenized
    然后统计token化的input_id的长度，再分别去85%和90%
    这个方法的唯一作用就是统计dialogue和summary的元素最大长度，然后计算得到max_source_length, max_target_length
    concatenate_datasets函数的作用就是将两个数据结构相同的数据，进行整合。
    """
    # 将train和test的数据集合并，并且对于dialogue进行truncation和batched处理
    # remove_columns，只保留编码后的词向量，减少消耗，便于训练
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    # 取最大长度的85%以更好地利用（np.percentile（A,B）B是一个百分数，取input_lenghts的85%长度，返回的长度，不是数组）
    max_source_length = int(np.percentile(input_lenghts, 85))
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 90))
    print(f"Max target length: {max_target_length}")
    return max_source_length, max_target_length


def preprocess_function(sample, padding="max_length"):
    """
    max_source_length, max_target_length,
    """
    # 为t5的输入添加前缀,并且讲对话进行拼接
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # 对话tokenize化。之前已经token过一次，那次是为了计算长度
    # 这里其实可以优化一下，两次token会损失一些性能
    # model_inputs和labels都是“input_ids”对应一个二维的数组
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # 使用“text_target”关键字参数标记目标。summary直接是一句话，无需拼接
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # 如果我们在这里padding，当我们想要忽略时，将标签中的所有tokenizer.pad_token_id替换为-100
    # padding in the loss.
    if padding == "max_length":
        # 即便利labels中的"input_ids"得到label，再遍历label数组，得到l。然后看l是不是padding的，是的话变为-100，不是的话不变
        # 然后再将l整合成数组，再放到"input_ids"这个key对应的value中
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    # model_inputs添加一个key，lables，也就是summary，也就是labels["input_ids"]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(model, tokenized_dataset):
    # 创建一个数据收集器，该收集器能够自动将原始数据转换为适合于Seq2Seq模型训练或评估的批次数据
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        # 这个参数是用于指定批次数据的长度应该是多少的倍数。如果指定为8，那么收集器将确保每个批次的输入和目标序列的长度都是8的倍数
        pad_to_multiple_of=8
    )

    # 设置训练参数，指定模型的输出路径、学习率、训练轮数、日志记录等。这些训练参数将决定训练的细节和效果
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # 设置为True时，会自动寻找适合的批次大小。
        auto_find_batch_size=True,
        learning_rate=1e-3,  # higher learning rate
        num_train_epochs=5,
        logging_dir=f"{output_dir}/logs",
        # 设置日志记录策略为"steps"，表示按照一定步数记录训练日志。
        logging_strategy="steps",
        # 设置每隔多少步记录一次训练日志
        logging_steps=500,
        # 设置模型保存策略为"no"，表示不保存模型
        save_strategy="no",
        report_to=["tensorboard"],
    )

    # 创建一个训练器对象，指定模型，训练参数，数据收集器，训练数据。准备开始训练
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False

    print("开始训练")
    trainer.train()

    print("训练完毕，保存数据")
    save_path = f"{output_dir}/results"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("数据保存完毕")


if __name__ == '__main__':
    try:
        resume_download_config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        # 获取编码token
        tokenizer = AutoTokenizer.from_pretrained(model_id, resume_download=resume_download_config)
        # 获取数据
        dataset = get_dataset()
        max_source_length, max_target_length = preprocess_dataset_truncate_padding(dataset)
        tokenized_dataset = preprocess_dataset(dataset)
        # 获取model
        model = get_model()
        train(model, tokenized_dataset)
        # 开始测试
        # LoRA_T5_Test.eval()
        sys.exit(0)
    except SystemExit:
        sys.exit(0)
