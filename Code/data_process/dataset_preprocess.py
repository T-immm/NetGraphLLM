import torch
import random
import re

def probabilistic_mask_with_positions(config, mask_token="[MASK]", mask_ratio=0.1):
    pattern = re.compile(r"(interface \S+)|(ip address \S+ \S+)")
    
    matches_with_positions = [(match.group(), match.span()) for match in pattern.finditer(config)]
    items_to_mask = [match for match, _ in matches_with_positions]
    
    max_mask_count = int(len(items_to_mask) * mask_ratio)
    
    if max_mask_count == 0:
        return config, []
    
    selection_indices = random.sample(range(len(matches_with_positions)), max_mask_count)
    selected_items_with_positions = [matches_with_positions[i] for i in selection_indices]


    labels_with_positions = []
    offset = 0  
    for item, (start, end) in selected_items_with_positions:
        labels_with_positions.append((item, start - offset))
        before = config[:start - offset]
        after = config[end - offset:]
        config = before + mask_token + after
        # 更新偏移量
        offset += (end - start) - len(mask_token)
    
    return config, labels_with_positions


def preprocess_function_pretrain_config(tokenizer, max_length=512):
    def preprocess_function(examples):
        desc_original = [f"{d}" for d in examples['config_desc']]
        # 随即掩码 并得到label
        masked_config, labels_with_positions = probabilistic_mask_with_positions(desc_original)
        
        
        desc_ids = tokenizer(masked_config,
                             add_special_tokens=True,
                             truncation=True,
                             max_length=max_length,
                             padding='max_length',
                             return_tensors='pt', )
        
        inputs = tokenizer(masked_config, return_tensors='pt')
        attention_mask = torch.zeros_like(inputs.input_ids)
        
        label_l = []
        for label, position in labels_with_positions:
            # 在需要预测的位置设置为1
            attention_mask[0, position] = 1
            label_l.append(label)
        labels = tokenizer(label_l, add_special_tokens=False)
            
        # 将attention mask赋给模型输入
        inputs['input_ids'] = desc_ids
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = labels
        return inputs

    return preprocess_function


def preprocess_function_original_understand(tokenizer, max_length=1200):
    def preprocess_function(examples):
        desc = [f"{d}" for d in examples['config_desc']]
        desc_ids = tokenizer(desc,
                             add_special_tokens=True,
                             truncation=True,
                             max_length=max_length,
                             padding='max_length',
                             return_tensors='pt', )
        return desc_ids

    return preprocess_function


def preprocess_function_original_update(tokenizer, max_length=512):
    def preprocess_function(examples):
        desc = [f"{d}" for d in examples['config_desc']]
        desc_ids = tokenizer(desc,
                             add_special_tokens=True,
                             truncation=True,
                             max_length=max_length,
                             padding='max_length',
                             return_tensors='pt', )
        return desc_ids

    return preprocess_function


def preprocess_function_generator_network_understand(tokenizer, ignore_index=-100, max_length=32):
    def preprocess_function(examples):
        """task_list = examples['task']
        
        prompts = []
        start = []
        end = []
        
        for task in task_list:
            start.append(task[0])
            end.append(task[1])
        
        print("start -----  {}".format(start[0 : 10]))
        print("end -----  {}".format(end[0 : 10]))
        
        for i in range(0, len(start)):
            text = "You are now an expert in dealing with network configuration issues, please answer: \
                What is the cost of the shortest path for Router {} to forward to Router {}?".format(start[i], end[i])
            prompts.append(text)"""
        
        query_list = examples['query']
        prompts = query_list
        

        completion = [f"{l}" for l in examples['labels']]
        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        # 将prompt输入
        model_inputs = tokenizer(prompts, add_special_tokens=False)
        labels = tokenizer(completion, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            # Add bos & eos token
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

            p_max_length = max_length - len(label_input_ids) - len(instruction)
            sample_input_ids = sample_input_ids[:p_max_length] + instruction

            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [ignore_index] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"][i]
            labels["input_ids"][i] = [ignore_index] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function




def preprocess_function_generator_config_update(tokenizer, ignore_index=-100, max_length=32):
    def preprocess_function(examples):
        query_list = examples['query']
        prompts = []
        
        for query in query_list:
            prompts.append(query)
            

        completion = [f"{l}" for l in examples['labels']]
        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        # 将prompt输入
        model_inputs = tokenizer(prompts, add_special_tokens=False)
        labels = tokenizer(completion, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            # Add bos & eos token
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]

            p_max_length = max_length - len(label_input_ids) - len(instruction)
            sample_input_ids = sample_input_ids[:p_max_length] + instruction

            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [ignore_index] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"][i]
            labels["input_ids"][i] = [ignore_index] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function























def preprocess_test_function_generator_network_understand(tokenizer, max_length=32):
    def preprocess_function(examples):
        
        """task_list = examples['task']
        
        prompts = []
        start = []
        end = []
        
        for task in task_list:
            start.append(task[0])
            end.append(task[1])
        
        for i in range(0, len(start)):
            text = "You are now an expert in dealing with network configuration issues, please answer: \
                What is the cost of the shortest path for Router {} to forward to Router {}?".format(start[i], end[i])
            prompts.append(text)"""
        
        prompts = examples['query']
        
        model_inputs = tokenizer(prompts, add_special_tokens=False)

        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i][
                                                          :max_length - len(instruction) - 1] + instruction

            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids

            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + [1] * len(
                sample_input_ids)
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])

        model_inputs['text_label'] = [f'{l}' for l in examples['labels']]
        return model_inputs

    return preprocess_function





def preprocess_test_function_generator_config_update(tokenizer, max_length=32):
    def preprocess_function(examples):
        
        query_list = examples['query']
        prompts = []
        
        for query in query_list:
            prompts.append(query)
        
        
        
        model_inputs = tokenizer(prompts, add_special_tokens=False)

        instruction = f"\n\n###\n\n"
        instruction = tokenizer.encode(instruction, add_special_tokens=False)

        batch_size = len(examples['node_ids'])

        for i in range(batch_size):
            sample_input_ids = [tokenizer.bos_token_id] + model_inputs["input_ids"][i][
                                                          :max_length - len(instruction) - 1] + instruction

            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids

            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + [1] * len(
                sample_input_ids)
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])

        model_inputs['text_label'] = [f'{l}' for l in examples['labels']]
        return model_inputs

    return preprocess_function
