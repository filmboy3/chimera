#!/usr/bin/env python3
"""
Sesame v2 Server-Side LLM Endpoint
Coaching dialogue generation with context-aware prompting
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import time
from typing import Dict, List, Optional
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoachingRequest(BaseModel):
    """Request model for coaching dialogue generation"""
    user_message: str
    fatigue_score: float  # 0-1
    focus_score: float    # 0-1
    rep_count: int
    target_reps: int
    form_errors: List[str] = []
    session_context: Dict = {}
    urgency: float = 0.5  # 0-1


class CoachingResponse(BaseModel):
    """Response model for coaching dialogue"""
    response_text: str
    confidence: float
    response_type: str  # "encouragement", "correction", "instruction", "answer"
    processing_time: float
    model_info: Dict


class CoachingLLMServer:
    """
    Server-side LLM for generating contextual coaching responses
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",  # Lightweight conversational model
        max_length: int = 150,
        temperature: float = 0.7,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Coaching templates and patterns
        self.coaching_templates = self._load_coaching_templates()
        
        logger.info(f"CoachingLLMServer initialized on {self.device}")
    
    def _load_coaching_templates(self) -> Dict[str, List[str]]:
        """Load coaching response templates"""
        
        return {
            "encouragement": [
                "Great work! Keep pushing through.",
                "You're doing amazing! {rep_count} down, {remaining} to go!",
                "Excellent form! Stay strong!",
                "Perfect! You've got this!",
                "Outstanding effort! Keep it up!"
            ],
            "fatigue_support": [
                "I can see you're working hard. Take a breath and focus on form.",
                "Feeling the burn? That means it's working! Stay controlled.",
                "You're stronger than you think. Push through this rep.",
                "Focus on your breathing. You can do this.",
                "Almost there! Don't give up now."
            ],
            "form_correction": {
                "knee_valgus": [
                    "Keep your knees tracking over your toes.",
                    "Push your knees out - don't let them cave in.",
                    "Focus on external rotation at the hips."
                ],
                "forward_lean": [
                    "Keep your chest up and core engaged.",
                    "Sit back into your hips more.",
                    "Maintain a proud chest throughout the movement."
                ],
                "insufficient_depth": [
                    "Go a little deeper - aim for hip crease below knee.",
                    "Full range of motion - get those hips down.",
                    "Challenge yourself with more depth."
                ]
            },
            "milestone_celebration": [
                "Halfway there! You're crushing it!",
                "Three-quarters done! Finish strong!",
                "Final rep coming up - make it count!",
                "Set complete! Excellent work!"
            ],
            "questions_answers": {
                "rep_count": "You've completed {rep_count} reps out of {target_reps}.",
                "form_check": "Your form looks {form_quality}. {specific_feedback}",
                "fatigue_check": "You're at {fatigue_level} fatigue. {fatigue_advice}",
                "remaining_reps": "You have {remaining} reps left. {encouragement}"
            }
        }
    
    def generate_coaching_response(self, request: CoachingRequest) -> CoachingResponse:
        """
        Generate contextual coaching response
        """
        start_time = time.time()
        
        try:
            # Analyze request context
            response_type = self._determine_response_type(request)
            
            # Generate context-rich prompt
            prompt = self._create_contextual_prompt(request, response_type)
            
            # Generate response
            if response_type in ["form_correction", "milestone_celebration"] and self._can_use_template(request):
                # Use template for common cases (faster)
                response_text = self._generate_template_response(request, response_type)
                confidence = 0.9
            else:
                # Use LLM for complex cases
                response_text = self._generate_llm_response(prompt)
                confidence = 0.8
            
            processing_time = time.time() - start_time
            
            return CoachingResponse(
                response_text=response_text,
                confidence=confidence,
                response_type=response_type,
                processing_time=processing_time,
                model_info={
                    "model_name": self.model_name,
                    "device": self.device,
                    "method": "template" if confidence > 0.85 else "llm"
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return CoachingResponse(
                response_text="Keep going! You're doing great!",
                confidence=0.5,
                response_type="fallback",
                processing_time=time.time() - start_time,
                model_info={"error": str(e)}
            )
    
    def _determine_response_type(self, request: CoachingRequest) -> str:
        """Determine the type of response needed"""
        
        # Check for form errors first
        if request.form_errors:
            return "form_correction"
        
        # Check for milestone celebrations
        progress = request.rep_count / request.target_reps if request.target_reps > 0 else 0
        if progress >= 1.0:
            return "milestone_celebration"
        elif progress >= 0.75:
            return "milestone_celebration"
        elif progress >= 0.5:
            return "milestone_celebration"
        
        # Check for questions
        question_indicators = ["how many", "how much", "am i", "is this", "what", "why", "when"]
        if any(indicator in request.user_message.lower() for indicator in question_indicators):
            return "answer"
        
        # Check fatigue level
        if request.fatigue_score > 0.7:
            return "fatigue_support"
        
        # Default to encouragement
        return "encouragement"
    
    def _create_contextual_prompt(self, request: CoachingRequest, response_type: str) -> str:
        """Create context-rich prompt for LLM"""
        
        # Base context
        context_parts = [
            f"You are an expert AI fitness coach providing real-time guidance during a squat workout.",
            f"User has completed {request.rep_count} out of {request.target_reps} reps.",
            f"Current fatigue level: {request.fatigue_score:.1f}/1.0 (higher = more tired)",
            f"Current focus level: {request.focus_score:.1f}/1.0 (higher = more focused)"
        ]
        
        # Add form error context
        if request.form_errors:
            errors_str = ", ".join(request.form_errors)
            context_parts.append(f"Detected form issues: {errors_str}")
        
        # Add response type guidance
        if response_type == "form_correction":
            context_parts.append("Provide specific, actionable form correction in a supportive tone.")
        elif response_type == "fatigue_support":
            context_parts.append("The user is fatigued. Provide encouraging but firm motivation.")
        elif response_type == "encouragement":
            context_parts.append("Provide positive encouragement to maintain motivation.")
        elif response_type == "answer":
            context_parts.append("Answer the user's question clearly and helpfully.")
        
        # User message
        if request.user_message.strip():
            context_parts.append(f"User said: '{request.user_message}'")
        
        context_parts.append("Generate a brief, motivating response (1-2 sentences):")
        
        return "\n".join(context_parts)
    
    def _can_use_template(self, request: CoachingRequest) -> bool:
        """Determine if we can use a template response (faster than LLM)"""
        
        # Use templates for common form corrections
        if request.form_errors and all(error in self.coaching_templates["form_correction"] for error in request.form_errors):
            return True
        
        # Use templates for milestone celebrations
        progress = request.rep_count / request.target_reps if request.target_reps > 0 else 0
        if progress in [0.5, 0.75, 1.0]:
            return True
        
        # Use templates for simple questions
        simple_questions = ["how many", "rep count", "reps left"]
        if any(q in request.user_message.lower() for q in simple_questions):
            return True
        
        return False
    
    def _generate_template_response(self, request: CoachingRequest, response_type: str) -> str:
        """Generate response using templates"""
        
        if response_type == "form_correction" and request.form_errors:
            # Use specific form correction templates
            error = request.form_errors[0]  # Handle first error
            if error in self.coaching_templates["form_correction"]:
                templates = self.coaching_templates["form_correction"][error]
                return templates[0]  # Use first template
        
        elif response_type == "milestone_celebration":
            progress = request.rep_count / request.target_reps if request.target_reps > 0 else 0
            if progress >= 1.0:
                return "Set complete! Excellent work!"
            elif progress >= 0.75:
                return "Three-quarters done! Finish strong!"
            elif progress >= 0.5:
                remaining = request.target_reps - request.rep_count
                return f"Halfway there! {remaining} more to go!"
        
        elif "how many" in request.user_message.lower():
            remaining = request.target_reps - request.rep_count
            return f"You have {remaining} reps remaining out of {request.target_reps}."
        
        # Fallback to encouragement
        return "Great work! Keep it up!"
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        
        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (after the prompt)
        response_text = generated_text[len(prompt):].strip()
        
        # Clean up response
        response_text = self._clean_response(response_text)
        
        return response_text
    
    def _clean_response(self, text: str) -> str:
        """Clean and validate generated response"""
        
        # Remove extra whitespace
        text = text.strip()
        
        # Limit length
        sentences = text.split('.')
        if len(sentences) > 2:
            text = '. '.join(sentences[:2]) + '.'
        
        # Remove incomplete sentences
        if text and not text.endswith(('.', '!', '?')):
            last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_punct > 0:
                text = text[:last_punct + 1]
        
        # Fallback if empty or too short
        if len(text) < 10:
            text = "Keep going! You're doing great!"
        
        return text


# FastAPI application
app = FastAPI(title="Sesame v2 Coaching LLM", version="1.0.0")

# Global model instance
coaching_llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize the coaching LLM on startup"""
    global coaching_llm
    
    model_name = os.getenv("COACHING_MODEL", "microsoft/DialoGPT-medium")
    coaching_llm = CoachingLLMServer(model_name=model_name)
    logger.info("Coaching LLM server started successfully")


@app.post("/generate_coaching_response", response_model=CoachingResponse)
async def generate_coaching_response(request: CoachingRequest):
    """
    Generate contextual coaching response
    """
    if coaching_llm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        response = coaching_llm.generate_coaching_response(request)
        return response
    except Exception as e:
        logger.error(f"Error in generate_coaching_response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": coaching_llm is not None,
        "device": coaching_llm.device if coaching_llm else "unknown"
    }


@app.get("/model_info")
async def model_info():
    """Get model information"""
    if coaching_llm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return {
        "model_name": coaching_llm.model_name,
        "device": coaching_llm.device,
        "max_length": coaching_llm.max_length,
        "temperature": coaching_llm.temperature
    }


def main():
    """Run the coaching LLM server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sesame v2 Coaching LLM Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Model name")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set environment variable for model
    os.environ["COACHING_MODEL"] = args.model
    
    # Run server
    uvicorn.run(
        "coaching_llm_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
