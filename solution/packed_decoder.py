"""Keep valid tokens packed between decoder layers in the private SFT route.

HF still builds the original embeddings, positions, masks and output object.
Its original decoder-layer math is used; pack once before layer zero and
restore once after the final norm. Unsupported/public consumers use HF unchanged.
"""
from contextvars import ContextVar
from functools import wraps
import torch

_packed_decoder = ContextVar('packed_qwen_decoder',default=None)


def _hooks(module):
    return bool(module._forward_hooks or module._forward_pre_hooks
                or module._backward_hooks or module._backward_pre_hooks
                or getattr(module,'_hf_hook',None) is not None)


def eligible(model, args, kwargs, state):
    from .precision import half_activations
    from .stream_head import _stream_request, _labels
    if (state is None or args or not model.training or not torch.is_grad_enabled()
            or not half_activations.get() or not _stream_request.get()
            or not isinstance(_labels.get(),torch.Tensor)
            or model.config.hidden_size != 1536 or model.config._attn_implementation != 'sdpa'
            or getattr(model,'gradient_checkpointing',False)
            or not torch.is_autocast_enabled('cuda') or torch.get_autocast_dtype('cuda') != torch.float16
            or state[1].numel() > 2048):
        return False
    permitted = {'input_ids','attention_mask','position_ids','past_key_values','inputs_embeds',
                 'use_cache','return_dict','output_hidden_states','output_attentions','num_items_in_batch',
                 'cache_position'}
    if set(kwargs)-permitted:
        return False
    ids = kwargs.get('input_ids')
    cache = kwargs.get('use_cache')
    cache = model.config.use_cache if cache is None else cache
    if (cache or kwargs.get('past_key_values') is not None or kwargs.get('inputs_embeds') is not None
            or kwargs.get('position_ids') is not None or kwargs.get('cache_position') is not None
            or kwargs.get('output_hidden_states',False) or kwargs.get('output_attentions',False)
            or getattr(model.config,'output_hidden_states',False) or getattr(model.config,'output_attentions',False)
            or kwargs.get('return_dict') is False or not model.config.return_dict
            or not isinstance(ids,torch.Tensor) or not ids.is_cuda or tuple(ids.shape) != state[0]):
        return False
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2Attention,Qwen2MLP,Qwen2RMSNorm
    from transformers.activations import SiLUActivation
    if (not getattr(Qwen2RMSNorm.forward,'_frozen_rms_patched',False)
            or not getattr(Qwen2Attention.forward,'_token_attention_patched',False)
            or type(model.norm) is not Qwen2RMSNorm or model.norm.weight.requires_grad
            or model.norm.weight.dtype != torch.float32 or 'forward' in model.norm.__dict__):
        return False
    embedding = model.embed_tokens
    if (embedding._forward_pre_hooks or embedding._backward_hooks or embedding._backward_pre_hooks
            or getattr(embedding,'_hf_hook',None) is not None
            or any(getattr(hook,'__module__','') != 'transformers.modeling_utils'
                   or 'enable_input_require_grads' not in getattr(hook,'__qualname__','')
                   for hook in embedding._forward_hooks.values())):
        return False
    if _hooks(model) or any(_hooks(m) for m in model.modules() if m is not model.embed_tokens):
        return False
    for layer in model.layers:
        attention = layer.self_attn
        if (type(layer) is not Qwen2DecoderLayer or type(attention) is not Qwen2Attention
                or type(layer.mlp) is not Qwen2MLP or attention.attention_dropout != 0.0
                or (attention.sliding_window is not None and attention.sliding_window < state[0][1])
                or 'forward' in layer.__dict__ or 'forward' in attention.__dict__
                or 'forward' in layer.mlp.__dict__ or 'forward' in layer.mlp.act_fn.__dict__
                or type(layer.mlp.act_fn) not in (torch.nn.SiLU,SiLUActivation)
                or getattr(layer.mlp.act_fn,'inplace',False)):
            return False
        for norm in (layer.input_layernorm,layer.post_attention_layernorm):
            if (type(norm) is not Qwen2RMSNorm or norm.weight.requires_grad
                    or norm.weight.dtype != torch.float32 or 'forward' in norm.__dict__):
                return False
        for projection in (attention.q_proj,attention.v_proj):
            if (not getattr(projection,'_rank_split_patched',False) or projection.disable_adapters
                    or projection.merged or len(projection.active_adapters) != 1
                    or not getattr(projection,'cast_input_dtype_enabled',True)):
                return False
            adapter = projection.active_adapters[0]
            if adapter not in projection.lora_A or adapter in projection.lora_variant:
                return False
            a,b,dropout = projection.lora_A[adapter],projection.lora_B[adapter],projection.lora_dropout[adapter]
            if (type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear
                    or a.bias is not None or b.bias is not None or tuple(a.weight.shape) != (16,1536)
                    or b.weight.shape[1] != 16 or b.weight.shape[0] > 1536
                    or any(w.dtype not in (torch.float16,torch.float32) or w.device != ids.device
                           for w in (a.weight,b.weight))
                    or type(dropout) not in (torch.nn.Dropout,torch.nn.Identity)
                    or getattr(dropout,'inplace',False)):
                return False
        projections = (attention.q_proj.get_base_layer(),attention.k_proj,
                       attention.v_proj.get_base_layer(),attention.o_proj,
                       layer.mlp.gate_proj,layer.mlp.up_proj,layer.mlp.down_proj)
        for projection in projections:
            if (type(projection) is not bnb.nn.Linear4bit or projection.weight.requires_grad
                    or projection.compute_dtype != torch.float16
                    or (projection.bias is not None and projection.bias.requires_grad)):
                return False
    return True


def patch_packed_decoder():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model,Qwen2DecoderLayer
    from .compact import _tokens,_pack,_restore
    original_model,original_layer = Qwen2Model.forward,Qwen2DecoderLayer.forward
    if getattr(original_model,'_packed_decoder_forward',False):
        return

    @wraps(original_model)
    def model_forward(self,*args,**kwargs):
        state = _tokens.get()
        if not eligible(self,args,kwargs,state):
            return original_model(self,*args,**kwargs)
        token = _packed_decoder.set(state)
        try:
            output = original_model(self,*args,**kwargs)
            hidden = output.last_hidden_state
            assert hidden.ndim == 2 and hidden.shape[0] == state[1].numel()
            output['last_hidden_state'] = _restore(hidden,state)
            return output
        finally:
            _packed_decoder.reset(token)

    @wraps(original_layer)
    def layer_forward(self,hidden_states,*args,**kwargs):
        state = _packed_decoder.get()
        if state is not None and hidden_states.ndim == 3:
            hidden_states = _pack(hidden_states,state)
        return original_layer(self,hidden_states,*args,**kwargs)

    model_forward._packed_decoder_forward = True
    Qwen2Model.forward = model_forward
    Qwen2DecoderLayer.forward = layer_forward
