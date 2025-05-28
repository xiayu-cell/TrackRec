from trl.commands.cli_utils import ScriptArguments, TrlParser
from dataclasses import dataclass

from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

@dataclass
class InputConfig:
    jsonl_path:str
    eval_path:str=None
    resume:str='false'


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, InputConfig))
    args, training_args, model_config, input_config = parser.parse_args_and_config()
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    print("max_seq_len:", training_args.max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    data_files = dict(train=input_config.jsonl_path)
    if input_config.eval_path:
        data_files.update(test=input_config.eval_path)
    print("loading dataset:", data_files)
    dataset = load_dataset('json',data_files=data_files)
    print("train dataset size:", len(dataset['train']))
    print("train data example:", dataset['train'][0])
    if input_config.eval_path:
        print("valid dataset size:", len(dataset['test']))
        print("valid data example:", dataset['test'][0])

    ################
    # Training
    ################
    peft_config = get_peft_config(model_config)
    if input_config.eval_path is not None:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            args=training_args,
            max_seq_length=training_args.max_seq_length,
            train_dataset=dataset[args.dataset_train_split],
            eval_dataset=dataset[args.dataset_test_split],
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            args=training_args,
            max_seq_length=training_args.max_seq_length,
            train_dataset=dataset[args.dataset_train_split],
            # eval_dataset=dataset[args.dataset_test_split],
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    # if input_config.resume !='false':
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()
    trainer.save_model(training_args.output_dir)
