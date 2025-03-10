import math
import copy
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.t5.modeling_t5 import (T5ForConditionalGeneration,
                                                T5PreTrainedModel,
                                                T5Stack,
                                                T5Block,
                                                T5EncoderModel)
from transformers import LogitsProcessor


class T5ForKnowledgeAugmentedGeneration(nn.Module):
    def __init__(self, args, entity_embeddings):
        super().__init__()
        self.response_generator = CustomT5ForConditionalGeneration.from_pretrained("t5-small", args=args)
        self.args = args


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        gold_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, 
    ):
        outputs = self.response_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        
        lm_logits = outputs[0]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            return (({'total_loss': loss},)+ outputs)
        else:
            return outputs

class KnowledgeGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.knowledge_generator = CustomT5ForConditionalGeneration.from_pretrained("t5-small", args=args)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        gold_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, # Belows are the additional inputs
    ):
        outputs = self.knowledge_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        
        lm_logits = outputs[0]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            return (({'total_loss': loss},)+ outputs)
        else:
            return outputs
        
        
class KnowledgePretrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.knowledge_pretrainer = CustomT5ForConditionalGeneration.from_pretrained("t5-small", args=args)

        for name, param in self.knowledge_pretrainer.named_parameters():
            if "new_embed" in name:
                param.requires_grad=True
            else:
                param.requires_grad=False
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        gold_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, # Belows are the additional inputs
    ):
        outputs = self.knowledge_pretrainer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss        

        return ({'total_loss': loss},)
    

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, args, ignore_mismatched_sizes=False):
        super().__init__(config)
        self.args = args
        
        self.new_embed = nn.Embedding(2+2*2*2*2, self.shared.weight.size(1)) # Head, Tail/ num_hops * #tokens * [Int, Rev] *2
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared, args, self.new_embed)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared, args, self.new_embed)


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None, args=None, new_embed=None):
        super().__init__(config, embed_tokens)
        self.args = args

        self.new_embed = new_embed

    def embed_input(self, input_ids):
        bz = input_ids.size(0)
        new_input_ids = input_ids.clone()
        new_input_ids[input_ids>=32100] = 2
        raw_embeds = self.embed_tokens(new_input_ids)
        
        blocked_indices = (input_ids>=32100).nonzero()
        for i, j in blocked_indices:
            raw_embeds[i, j,:] = self.new_embed(input_ids[i, j]-32100)

        return raw_embeds
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        graph_inputs=None
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input(input_ids)
            return super().forward(
                None,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                inputs_embeds,
                head_mask,
                cross_attn_head_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
        
        else:
            return super().forward(
                None,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                inputs_embeds,
                head_mask,
                cross_attn_head_mask,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )


class GraphConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn, num_beams, args):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.args = args
        
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -math.inf)
        scores= scores.log_softmax(dim=-1)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.size(-1))):
            for beam_id, sent in enumerate(beam_sent):
                temp = self._prefix_allowed_tokens_fn(batch_id, sent)
                if temp is None:
                    mask[batch_id*self._num_beams + beam_id, :] = 0
                elif len(temp) != 0:
                    prefix_allowed_tokens, prefix_allowed_token_values = temp[0], temp[1]
                    mask[batch_id*self._num_beams + beam_id, prefix_allowed_tokens] = torch.tensor(prefix_allowed_token_values, dtype=torch.float32, device=scores.device).log_softmax(dim=-1)

        changed_logits = scores*self.args.lm_weight + mask*(1-self.args.lm_weight)
        return changed_logits
        