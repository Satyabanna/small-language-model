# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from dataclasses import dataclass

# --- Configuration Section ---
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

# --- Model Architecture (same as training) ---
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Helper Functions ---
def clean_response(text):
    """Clean and format the model response"""
    # Remove special tokens
    text = text.replace('<bos>', '').replace('<eos>', '').strip()
    
    # Find doctor's response (after the patient query)
    if 'Doctor:' in text:
        # Extract only the doctor's part
        doctor_start = text.find('Doctor:')
        if doctor_start != -1:
            doctor_response = text[doctor_start + 7:].strip()  # +7 for "Doctor:"
            
            # Clean up the response
            doctor_response = doctor_response.split('<eos>')[0].strip()
            doctor_response = doctor_response.split('Patient:')[0].strip()  # Stop at next patient
            
            # Remove repetitive patterns and incomplete sentences
            sentences = doctor_response.split('.')
            clean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and sentence not in clean_sentences:  # Avoid very short or duplicate sentences
                    clean_sentences.append(sentence)
                if len(clean_sentences) >= 3:  # Limit to 3 sentences for clarity
                    break
            
            if clean_sentences:
                result = '. '.join(clean_sentences)
                if not result.endswith('.'):
                    result += '.'
                return result
    
    return "I apologize, but I couldn't generate a clear response. Please try rephrasing your question."

def medical_chat():
    """Interactive medical chat interface"""
    print("🏥 Medical AI Assistant")
    print("=" * 50)
    print("Type your medical questions or symptoms.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    
    # Initialize model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Same configuration as training
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True
    )
    
    # Load the model
    model = GPT(config)
    best_model_params_path = "best_model_params.pt"
    
    try:
        model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: best_model_params.pt not found. Please run training first.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    print("\n🤖 Medical Assistant is ready! How can I help you today?\n")
    
    while True:
        try:
            # Get user input
            patient_query = input("👤 Patient: ").strip()
            
            # Check for exit commands
            if patient_query.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Thank you for using the Medical AI Assistant. Stay healthy!")
                break
            
            if not patient_query:
                print("Please enter your medical question or concern.")
                continue
            
            # Format input similar to training data with better prompt engineering
            # Add more context to help the model understand the medical consultation format
            input_text = f"<bos> Medical consultation between patient and doctor Patient: {patient_query} Doctor:"
            
            # Tokenize input
            input_ids = enc.encode(input_text)
            
            # Truncate if too long to fit in context window
            if len(input_ids) > config.block_size - 50:  # Leave room for generation
                input_ids = input_ids[-(config.block_size - 50):]
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            print("\n🤔 Analyzing your symptoms...")
            
            # Generate response with better parameters
            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=80,   # Shorter responses for better quality
                    temperature=0.7,     # Less randomness for more coherent responses
                    top_k=40            # More focused vocabulary
                )
            
            # Decode response
            output_text = enc.decode(output[0].tolist())
            
            # Clean and extract doctor's response
            doctor_response = clean_response(output_text)
            
            # Additional check for response quality
            if len(doctor_response) < 20 or "I apologize" in doctor_response:
                # Try with different parameters if response is poor
                print("🔄 Generating alternative response...")
                with torch.no_grad():
                    output = model.generate(
                        input_tensor,
                        max_new_tokens=60,
                        temperature=0.6,  # Even more focused
                        top_k=30
                    )
                output_text = enc.decode(output[0].tolist())
                doctor_response = clean_response(output_text)
            
            # Display response
            if doctor_response and len(doctor_response) > 10:
                print(f"\n🩺 Doctor: {doctor_response}")
            else:
                print(f"\n🩺 Doctor: I understand your concern about '{patient_query}'. For this type of symptom, I recommend consulting with a healthcare professional who can provide a proper examination and diagnosis. In the meantime, monitor your symptoms and seek immediate care if they worsen.")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Take care!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different question.\n")

if __name__ == "__main__":
    medical_chat()