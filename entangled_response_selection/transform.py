class SelectionConcatTransform(object):
    def __init__(self, tokenizer, max_len, max_num_contexts):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_num_contexts = max_num_contexts

    def __call__(self, contexts, responses):
        # pad contexts
        contexts = contexts[-self.max_num_contexts:]
        diff = self.max_num_contexts - len(contexts)
        if diff > 0:
            contexts = ['this is a padding']*diff + contexts

        ret_input_ids = []
        ret_input_masks = []
        ret_segment_ids = []
        
        for response in responses:
            tokenized_dict = self.tokenizer(contexts+[response], [response]*(len(contexts)+1), padding='max_length', max_length=self.max_len, truncation='longest_first')
            #tokenized_dict = self.tokenizer(contexts, [response]*len(contexts), padding='max_length', max_length=self.max_len, truncation='longest_first')
            input_ids, input_masks, input_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
            ret_input_ids.append(input_ids)
            ret_input_masks.append(input_masks)
            ret_segment_ids.append(input_segment_ids)
        
        return ret_input_ids, ret_input_masks, ret_segment_ids
