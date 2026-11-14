"""Fusion controls from the frozen experiment."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .model import PacketHierarchicalFusionNet
MODES=("full", "behavior_only", "late_fusion", "packet_concat", "competitive_gate")

class FusionComparison(nn.Module):
    def __init__(self,classes,mode,dim=128,dropout=0.1):
        super().__init__();assert mode in MODES and mode!='full';self.mode=mode
        # Instantiate the same backbone first: common encoder/GRU initial values
        # match those of the full model for the same seed.
        self.base=PacketHierarchicalFusionNet(classes,dim=dim,dropout=dropout)
        if mode=='behavior_only':
            self.base.byte_encoder=None;self.base.byte_head=None;self.base.packet_gate=None
        if mode=='packet_concat':
            self.base.packet_gate=None;self.concat_projection=nn.Linear(dim*2,dim)
    def aggregate(self,x,mask,count):
        lengths=count.clamp(min=1,max=x.shape[1]).cpu();packed=pack_padded_sequence(x*mask[...,None],lengths,batch_first=True,enforce_sorted=False)
        output,_=self.base.packet_gru(packed);seq,_=pad_packed_sequence(output,batch_first=True,total_length=x.shape[1])
        scores=self.base.packet_attention(seq).squeeze(-1).masked_fill(~mask,torch.finfo(seq.dtype).min)
        empty=~mask.any(1)
        if empty.any():scores=scores.clone();scores[empty,0]=0
        return (seq*scores.softmax(1)[...,None]).sum(1)
    def gate(self,b,t):return torch.sigmoid(self.base.packet_gate(torch.cat([b,t,(b-t).abs()],dim=-1)))
    def forward(self,byte,features,count):
        mask=torch.arange(byte.shape[1],device=byte.device)[None,:]<count.clamp(min=0,max=byte.shape[1])[:,None]
        t=self.base.behavior_projection(features.float());t=F.gelu(self.base.behavior_convolution(F.pad(t.transpose(1,2),(2,0))).transpose(1,2))*mask[...,None]
        g=None
        if self.mode=='behavior_only':z=self.aggregate(self.base.packet_norm(t),mask,count)
        else:
            b=self.base.byte_encoder(byte,features[...,0])*mask[...,None]
            if self.mode=='late_fusion':
                # Shared aggregator keeps parameter capacity controlled; evaluated
                # twice, which is recorded as an additional compute cost.
                bf=self.aggregate(b,mask,count);tf=self.aggregate(t,mask,count);g=self.gate(bf,tf);z=self.base.packet_norm(bf+g*tf)
            else:
                if self.mode=='packet_concat':x=self.concat_projection(torch.cat([b,t],dim=-1))
                else:g=self.gate(b,t);x=g*b+(1-g)*t
                z=self.aggregate(self.base.packet_norm(x),mask,count)
        return dict(logits=self.base.fusion_head(self.base.dropout(z)),gate=g)
