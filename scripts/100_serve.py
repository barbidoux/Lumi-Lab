#!/usr/bin/env python3
"""
Inference script to serve the trained model.
Supports both interactive CLI mode and API server mode.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
import sentencepiece as spm

from utils.model_utils import load_pretrained_model
from utils.tokenizer_utils import SentencePieceTokenizerWrapper




class GenerationRequest(BaseModel):
    """Request model for generation API."""
    prompt: str
    max_new_tokens: Optional[int] = 96
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = True
    template: Optional[str] = "chatml"


class GenerationResponse(BaseModel):
    """Response model for generation API."""
    response: str
    prompt: str
    generation_config: Dict


class BatchGenerationRequest(BaseModel):
    """Request model for batch generation API."""
    prompts: List[str]
    max_new_tokens: Optional[int] = 96
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.1
    do_sample: Optional[bool] = True
    template: Optional[str] = "chatml"


class BatchGenerationResponse(BaseModel):
    """Response model for batch generation API."""
    responses: List[str]
    prompts: List[str]
    generation_config: Dict


class ModelServer:
    """Model server with text generation."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize server with model and tokenizer.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer (optional)
        """
        print(f"Loading model from {model_path}...")
        self.model = load_pretrained_model(model_path)
        self.model.eval()
        
        # Load tokenizer
        if tokenizer_path:
            # Handle SentencePiece tokenizer files
            if tokenizer_path.endswith('.model'):
                # Load SentencePiece model directly
                self.tokenizer = SentencePieceTokenizerWrapper(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Inference optimizations
        self.device = next(self.model.parameters()).device

        # Enable memory-efficient inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Clear cache
            torch.cuda.empty_cache()

        print(f"Model loaded on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def format_prompt(self, prompt: str, template: str = "chatml") -> str:
        """
        Format prompt according to specified template.
        
        Args:
            prompt: User prompt
            template: Template to use
            
        Returns:
            Formatted prompt
        """
        if template == "chatml":
            return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        elif template == "chat":
            return f"Human: {prompt}\n\nAssistant: "
        elif template == "instruct":
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            # Raw template - no formatting
            return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        template: str = "chatml"
    ) -> tuple[str, str]:
        """
        Generate response from prompt.
        
        Args:
            prompt: Prompt utilisateur
            max_new_tokens: Nombre maximum de nouveaux tokens
            temperature: Generation temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            do_sample: Use sampling
            template: Template de formatage
            
        Returns:
            Tuple (formatted_prompt, generated_response)
        """
        # Format prompt
        formatted_prompt = self.format_prompt(prompt, template)
        
        # Tokenization
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation configuration with default parameters
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3
        }
        
        # Only add early_stopping for beam search (not sampling)
        if not do_sample:
            generation_config["early_stopping"] = True
        
        # Generation with memory optimization
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **generation_config
                )
            except torch.cuda.OutOfMemoryError:
                # Clear cache and retry
                torch.cuda.empty_cache()
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **generation_config
                )
        
        # ROBUST: Extract only newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]  # Only new tokens
        generated_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return formatted_prompt, generated_response


def create_app(model_server: ModelServer) -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="Lumi Model Server",
        description="API to interact with trained Lumi model",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Root endpoint with model information."""
        return {
            "message": "Lumi Model Server",
            "model_device": str(model_server.device),
            "model_parameters": sum(p.numel() for p in model_server.model.parameters()),
            "available_templates": ["chatml", "chat", "instruct", "raw"],
            "endpoints": {
                "generate": "/generate - Single text generation",
                "batch_generate": "/generate/batch - Batch text generation",
                "stream_generate": "/generate/stream - Streaming text generation",
                "health": "/health - Server status"
            }
        }
    
    @app.get("/health")
    async def health():
        """Server health endpoint."""
        return {"status": "healthy", "device": str(model_server.device)}
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate(request: GenerationRequest):
        """
        Text generation endpoint.
        
        Args:
            request: Generation request
            
        Returns:
            Generated response
        """
        try:
            formatted_prompt, response = model_server.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=getattr(request, 'top_k', 50),
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                template=request.template
            )
            
            return GenerationResponse(
                response=response,
                prompt=request.prompt,
                generation_config={
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": getattr(request, 'top_k', 50),
                    "repetition_penalty": request.repetition_penalty,
                    "template": request.template
                }
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    @app.post("/generate/batch", response_model=BatchGenerationResponse)
    async def generate_batch(request: BatchGenerationRequest):
        """
        Batch text generation endpoint.

        Args:
            request: Batch generation request

        Returns:
            Batch generated responses
        """
        try:
            responses = []
            for prompt in request.prompts:
                _, response = model_server.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    template=request.template
                )
                responses.append(response)

            return BatchGenerationResponse(
                responses=responses,
                prompts=request.prompts,
                generation_config={
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "template": request.template
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch generation error: {str(e)}")

    @app.post("/generate/stream")
    async def generate_stream(request: GenerationRequest):
        """
        Streaming text generation endpoint.
        Returns tokens as they are generated.
        """
        def token_generator():
            try:
                formatted_prompt = model_server.format_prompt(request.prompt, request.template)
                inputs = model_server.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )
                inputs = {k: v.to(model_server.device) for k, v in inputs.items()}
                input_length = inputs['input_ids'].shape[1]

                # Generate token by token
                current_ids = inputs['input_ids']
                for _ in range(request.max_new_tokens):
                    with torch.no_grad():
                        outputs = model_server.model(current_ids)
                        logits = outputs.logits[0, -1, :]

                        # Apply temperature and sampling
                        if request.do_sample:
                            logits = logits / request.temperature
                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)

                        # Decode and yield token
                        token_text = model_server.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                        yield f"data: {json.dumps({'token': token_text})}\n\n"

                        # Update sequence
                        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

                        # Check for EOS
                        if next_token.item() == model_server.tokenizer.eos_token_id:
                            break

                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(token_generator(), media_type="text/plain")

    return app


def interactive_mode(model_server: ModelServer, args):
    """Interactive CLI mode to chat with the model."""
    
    print("🤖 Lumi Interactive Mode")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Template: {args.template}")
    print(f"Parameters: temp={args.temperature}, top_p={args.top_p}, top_k={getattr(args, 'top_k', 50)}")
    print("Type 'exit', 'quit' or Ctrl+C to quit")
    print("=" * 50)
    
    try:
        while True:
            # Saisie utilisateur
            try:
                user_input = input("\n👤 Vous: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break
            
            # Special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("🤖 Lumi: ", end="", flush=True)
            
            try:
                _, response = model_server.generate(
                    prompt=user_input,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=getattr(args, 'top_k', 50),
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.do_sample,
                    template=args.template
                )
                
                print(response)
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")


def main():
    parser = argparse.ArgumentParser(description="Lumi inference server")
    
    # Arguments principaux
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/spm32k.model",
                       help="Path to SentencePiece tokenizer (default: data/tokenizer/spm32k.model)")
    parser.add_argument("--mode", type=str, default="interactive", 
                       choices=["interactive", "api"],
                       help="Execution mode: interactive or api")
    
    # Generation parameters with improved defaults
    parser.add_argument("--max_new_tokens", type=int, default=96,
                       help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling")
    parser.add_argument("--template", type=str, default="chatml",
                       choices=["chatml", "chat", "instruct", "raw"],
                       help="Prompt formatting template")
    
    # API parameters
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="IP address for API server")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for API server")
    
    args = parser.parse_args()
    
    # Model verification
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    # Initialize model server
    try:
        model_server = ModelServer(args.model_path, args.tokenizer_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Launch according to mode
    if args.mode == "interactive":
        interactive_mode(model_server, args)

    elif args.mode == "api":
        print(f"🚀 Starting API server on {args.host}:{args.port}")
        app = create_app(model_server)
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )


if __name__ == "__main__":
    main()