#!/usr/bin/env python3
"""
Sesame v2 Intent Recognition - MobileBERT Intent Classification
Fast, on-device intent classification for [direct_command], [question_for_coach], [ambient_chatter]
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class MobileBERTIntentClassifier(nn.Module):
    """
    Lightweight MobileBERT-based intent classifier
    Optimized for on-device inference with Core ML conversion
    """
    
    def __init__(
        self,
        model_name: str = "google/mobilebert-uncased",
        num_intents: int = 3,
        max_length: int = 64,  # Short utterances for fitness coaching
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_length = max_length
        self.num_intents = num_intents
        
        # Load MobileBERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze most BERT layers for efficiency (fine-tune only last 2 layers)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 encoder layers
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_intents)
        )
        
        # Intent labels
        self.intent_labels = [
            'direct_command',      # "start workout", "stop", "pause"
            'question_for_coach',  # "how many more reps?", "am I doing this right?"
            'ambient_chatter'      # Background conversation, not directed at coach
        ]
        
        # Command patterns for direct classification
        self.command_patterns = {
            'direct_command': [
                'start', 'stop', 'pause', 'resume', 'go', 'begin', 'end', 'quit',
                'faster', 'slower', 'harder', 'easier', 'next', 'skip', 'repeat'
            ],
            'question_for_coach': [
                'how many', 'how much', 'am i', 'is this', 'should i', 'what',
                'why', 'when', 'where', 'correct', 'right', 'wrong', 'good', 'better'
            ]
        }
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for intent classification
        
        Args:
            input_ids: (batch_size, seq_len) - Tokenized input
            attention_mask: (batch_size, seq_len) - Attention mask
            
        Returns:
            logits: (batch_size, num_intents) - Intent classification logits
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_intent(self, text: str, use_patterns: bool = True) -> Dict[str, float]:
        """
        Predict intent for a single text input
        
        Args:
            text: Input text
            use_patterns: Whether to use pattern matching as fallback
            
        Returns:
            Dictionary with intent probabilities
        """
        self.eval()
        
        # Quick pattern matching for common cases (faster than BERT)
        if use_patterns:
            pattern_intent = self._pattern_match_intent(text)
            if pattern_intent:
                return {
                    'intent': pattern_intent,
                    'confidence': 0.9,  # High confidence for pattern matches
                    'method': 'pattern_matching'
                }
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = self.forward(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(logits, dim=-1).squeeze()
            
            # Get predicted intent
            predicted_idx = torch.argmax(probabilities).item()
            predicted_intent = self.intent_labels[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            return {
                'intent': predicted_intent,
                'confidence': confidence,
                'method': 'bert_classification',
                'all_probabilities': {
                    label: prob.item() for label, prob in zip(self.intent_labels, probabilities)
                }
            }
    
    def _pattern_match_intent(self, text: str) -> Optional[str]:
        """
        Fast pattern matching for common intents
        Returns intent if confident match, None otherwise
        """
        text_lower = text.lower()
        
        # Check for direct commands
        for pattern in self.command_patterns['direct_command']:
            if pattern in text_lower:
                return 'direct_command'
        
        # Check for questions
        for pattern in self.command_patterns['question_for_coach']:
            if pattern in text_lower:
                return 'question_for_coach'
        
        # Check for question words at start
        if text_lower.strip().startswith(('how', 'what', 'why', 'when', 'where', 'is', 'am', 'should', 'can', 'will')):
            return 'question_for_coach'
        
        # Check for imperative commands (short utterances)
        if len(text.split()) <= 3 and any(word in text_lower for word in ['start', 'stop', 'go', 'pause']):
            return 'direct_command'
        
        return None
    
    def get_model_size(self) -> Dict[str, int]:
        """Calculate model size for deployment planning"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


class IntentRecognitionPipeline:
    """
    Complete intent recognition pipeline with caching and optimization
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        use_cache: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            self.model = MobileBERTIntentClassifier()
        
        self.model.eval()
        
        # Response cache for repeated utterances
        self.cache = {} if use_cache else None
        self.cache_hits = 0
        self.total_queries = 0
        
        print(f"IntentRecognitionPipeline initialized with {self.model.get_model_size()['model_size_mb']:.1f}MB model")
    
    def _load_model(self, model_path: str) -> MobileBERTIntentClassifier:
        """Load fine-tuned model"""
        model = MobileBERTIntentClassifier()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def classify_intent(self, text: str) -> Dict[str, any]:
        """
        Classify intent with caching and confidence filtering
        
        Args:
            text: Input utterance
            
        Returns:
            Intent classification result with gating decision
        """
        self.total_queries += 1
        
        # Normalize text
        text_normalized = text.strip().lower()
        
        # Check cache
        if self.cache and text_normalized in self.cache:
            self.cache_hits += 1
            return self.cache[text_normalized]
        
        # Classify intent
        result = self.model.predict_intent(text)
        
        # Apply confidence gating
        should_activate_llm = (
            result['confidence'] >= self.confidence_threshold and
            result['intent'] in ['direct_command', 'question_for_coach']
        )
        
        # Enhanced result with gating decision
        enhanced_result = {
            **result,
            'should_activate_llm': should_activate_llm,
            'gating_reason': self._get_gating_reason(result),
            'processing_time': 0.0  # Would be measured in real implementation
        }
        
        # Cache result
        if self.cache:
            self.cache[text_normalized] = enhanced_result
        
        return enhanced_result
    
    def _get_gating_reason(self, result: Dict) -> str:
        """Get human-readable reason for LLM gating decision"""
        
        if result['intent'] == 'ambient_chatter':
            return "Classified as ambient chatter - not directed at coach"
        elif result['confidence'] < self.confidence_threshold:
            return f"Low confidence ({result['confidence']:.2f}) - below threshold ({self.confidence_threshold})"
        elif result['intent'] in ['direct_command', 'question_for_coach']:
            return f"High confidence {result['intent']} - activating LLM"
        else:
            return "Unknown gating condition"
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get caching performance statistics"""
        if not self.cache or self.total_queries == 0:
            return {'cache_hit_rate': 0.0, 'total_queries': self.total_queries}
        
        return {
            'cache_hit_rate': self.cache_hits / self.total_queries,
            'total_queries': self.total_queries,
            'cache_size': len(self.cache)
        }


class IntentDataGenerator:
    """
    Generate synthetic training data for intent classification
    """
    
    def __init__(self):
        # Template patterns for each intent
        self.templates = {
            'direct_command': [
                "start workout",
                "stop the exercise", 
                "pause for a moment",
                "resume training",
                "go faster",
                "slow down",
                "begin the session",
                "end workout",
                "start squats",
                "stop counting",
                "next exercise",
                "repeat that",
                "skip this one"
            ],
            'question_for_coach': [
                "how many more reps?",
                "am I doing this right?",
                "is my form correct?",
                "should I go deeper?",
                "what's my rep count?",
                "how much longer?",
                "am I going too fast?",
                "is this the right depth?",
                "why does this hurt?",
                "when should I rest?",
                "how many sets left?",
                "what comes next?",
                "can I take a break?"
            ],
            'ambient_chatter': [
                "hey did you see the game last night",
                "what time is dinner",
                "the weather is nice today",
                "I need to pick up groceries",
                "my phone is ringing",
                "someone's at the door",
                "the music is too loud",
                "I forgot to call mom",
                "traffic was terrible",
                "this song is great",
                "I'm hungry",
                "what's on TV tonight"
            ]
        }
    
    def generate_training_data(self, samples_per_intent: int = 1000) -> Dict[str, List[Dict]]:
        """Generate synthetic training data with variations"""
        
        training_data = {intent: [] for intent in self.templates.keys()}
        
        for intent, templates in self.templates.items():
            for i in range(samples_per_intent):
                # Select random template
                template = np.random.choice(templates)
                
                # Apply variations
                varied_text = self._apply_variations(template)
                
                training_data[intent].append({
                    'text': varied_text,
                    'intent': intent,
                    'template_id': templates.index(template) if template in templates else -1
                })
        
        return training_data
    
    def _apply_variations(self, text: str) -> str:
        """Apply linguistic variations to template text"""
        
        variations = [
            text,  # Original
            text.capitalize(),
            text.upper(),
            f"um {text}",
            f"{text} please",
            f"can you {text}",
            f"I want to {text}",
            text.replace("I", "i"),  # Casual typing
            text + "?",  # Add question mark
            text + "...",  # Add ellipsis
        ]
        
        return np.random.choice(variations)
    
    def save_training_data(self, output_path: str, samples_per_intent: int = 1000):
        """Generate and save training data"""
        
        data = self.generate_training_data(samples_per_intent)
        
        # Flatten into single list with labels
        training_samples = []
        for intent, samples in data.items():
            training_samples.extend(samples)
        
        # Shuffle
        np.random.shuffle(training_samples)
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(training_samples, f, indent=2)
        
        print(f"Generated {len(training_samples)} training samples saved to {output_file}")
        
        # Print distribution
        intent_counts = {}
        for sample in training_samples:
            intent = sample['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("Intent distribution:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} samples")


def main():
    """Test intent recognition pipeline"""
    
    # Create pipeline
    pipeline = IntentRecognitionPipeline()
    
    # Test utterances
    test_utterances = [
        "start workout",
        "how many more reps do I have?",
        "hey did you see that movie",
        "am I doing this right?",
        "stop the exercise",
        "what time is it",
        "pause for a second",
        "is my form correct?",
        "I need to call my mom"
    ]
    
    print("Testing Intent Recognition Pipeline:")
    print("=" * 50)
    
    for utterance in test_utterances:
        result = pipeline.classify_intent(utterance)
        
        print(f"Text: '{utterance}'")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.3f})")
        print(f"Activate LLM: {result['should_activate_llm']}")
        print(f"Reason: {result['gating_reason']}")
        print("-" * 30)
    
    # Print cache stats
    cache_stats = pipeline.get_cache_stats()
    print(f"\nCache Performance:")
    print(f"Hit Rate: {cache_stats['cache_hit_rate']:.1%}")
    print(f"Total Queries: {cache_stats['total_queries']}")
    
    # Generate training data
    print("\nGenerating training data...")
    generator = IntentDataGenerator()
    generator.save_training_data("./training_data/intent_training.json", samples_per_intent=100)


if __name__ == "__main__":
    main()
